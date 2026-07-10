from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Callable
from urllib.parse import urlparse

import torch

from .agents import (
    HeuristicAgent,
    HybridAgent,
    NeuralAgent,
    RandomAgent,
    RolloutAgent,
)
from .game import AzulGame, COLORS, decode_action


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).with_name("web_static")


def _checkpoint(relative: str) -> Path:
    return PROJECT_ROOT / relative


def _neural(relative: str):
    return NeuralAgent(_checkpoint(relative))


def _hybrid(relative: str):
    return HybridAgent(_checkpoint(relative))


def _rollout(relative: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RolloutAgent(
        _checkpoint(relative), device=device, top_k=4,
        search_from_round=5, objective="own",
    )


OPPONENTS: dict[str, tuple[str, str, Callable[[], object], str | None]] = {
    "competitive_hybrid": (
        "Competitive hybrid", "Strongest measured match-play opponent",
        lambda: _hybrid("checkpoints/best_competitive.pt"),
        "checkpoints/best_competitive.pt",
    ),
    "score_neural": (
        "Score champion", "Raw neural policy with the highest self-play score",
        lambda: _neural("checkpoints/best_score.pt"),
        "checkpoints/best_score.pt",
    ),
    "score_rollout": (
        "Score champion + search", "Score policy with final-round rollout search; slower but sharper",
        lambda: _rollout("checkpoints/best_score.pt"),
        "checkpoints/best_score.pt",
    ),
    "score_hybrid": (
        "Score hybrid", "Score policy constrained by tactical guardrails",
        lambda: _hybrid("checkpoints/best_score.pt"),
        "checkpoints/best_score.pt",
    ),
    "competitive_neural": (
        "Competitive neural", "Raw neural policy trained for head-to-head play",
        lambda: _neural("checkpoints/best_competitive.pt"),
        "checkpoints/best_competitive.pt",
    ),
    "multi_outer": (
        "Experimental multi-star", "Experimental policy that completes more outer stars",
        lambda: _neural("checkpoints/multi_outer_recover_v3/latest.pt"),
        "checkpoints/multi_outer_recover_v3/latest.pt",
    ),
    "heuristic": (
        "Strategic heuristic", "Fast handcrafted strategic baseline",
        lambda: HeuristicAgent(17), None,
    ),
    "random": (
        "Random", "Gentle introduction; selects uniformly from legal moves",
        lambda: RandomAgent(17), None,
    ),
}


def available_opponents() -> list[dict]:
    result = []
    for key, (name, description, _, checkpoint) in OPPONENTS.items():
        available = checkpoint is None or _checkpoint(checkpoint).exists()
        result.append({
            "id": key,
            "name": name,
            "description": description,
            "available": available,
        })
    return result


def _player_payload(player, index: int) -> dict:
    return {
        "index": index,
        "name": "You" if index == 0 else "AI",
        "score": player.score,
        "inventory": dict(zip(COLORS, player.inventory)),
        "stored": dict(zip(COLORS, player.stored)),
        "outer": [
            {"color": COLORS[color], "slots": list(star)}
            for color, star in enumerate(player.outer)
        ],
        "center": [COLORS[color] if color >= 0 else None for color in player.center],
        "claimed": [
            {"kind": kind, "index": index}
            for kind, index in sorted(player.claimed)
        ],
        "passed": player.passed,
        "tiles_on_board": player.board_tile_count(),
    }


def _action_payload(game: AzulGame, action_id: int) -> dict:
    action = decode_action(action_id)
    return {
        "id": action_id,
        "description": game.action_description(action_id),
        "kind": action.kind,
        "source": action.source,
        "color": COLORS[action.color] if action.color >= 0 else None,
        "star": COLORS[action.star] if 0 <= action.star < 6 else ("center" if action.star == 6 else None),
        "cost": action.slot + 1 if action.slot >= 0 else None,
        "natural": action.natural if action.natural >= 0 else None,
    }


class GameSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.game: AzulGame | None = None
        self.agent = None
        self.opponent_id = "competitive_hybrid"
        self.last_ai_actions: list[str] = []

    def new_game(self, opponent_id: str, seed: int | None = None) -> dict:
        if opponent_id not in OPPONENTS:
            raise ValueError("Unknown opponent")
        name, _, factory, checkpoint = OPPONENTS[opponent_id]
        if checkpoint and not _checkpoint(checkpoint).exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        with self.lock:
            self.agent = factory()
            self.opponent_id = opponent_id
            self.game = AzulGame(seed=seed)
            self.last_ai_actions = []
            self._play_ai_turns()
            return self.payload()

    def act(self, action_id: int) -> dict:
        with self.lock:
            if self.game is None:
                raise ValueError("Start a game first")
            if self.game.done:
                raise ValueError("The game is already over")
            if self.game.current_player != 0:
                raise ValueError("It is the AI's turn")
            if action_id not in self.game.legal_actions():
                raise ValueError("That action is no longer legal")
            self.game.step(action_id)
            self._play_ai_turns()
            return self.payload()

    def _play_ai_turns(self) -> None:
        self.last_ai_actions = []
        guard = 0
        while self.game is not None and not self.game.done and self.game.current_player == 1:
            action = self.agent.choose(self.game)
            self.last_ai_actions.append(self.game.action_description(action))
            self.game.step(action)
            guard += 1
            if guard > 100:
                raise RuntimeError("AI turn loop exceeded safety limit")

    def payload(self) -> dict:
        if self.game is None:
            return {
                "started": False,
                "opponents": available_opponents(),
            }
        game = self.game
        legal = game.legal_actions() if not game.done and game.current_player == 0 else []
        winner = None
        if game.done:
            if game.players[0].score > game.players[1].score:
                winner = "human"
            elif game.players[1].score > game.players[0].score:
                winner = "ai"
            else:
                winner = "tie"
        return {
            "started": True,
            "opponents": available_opponents(),
            "opponent_id": self.opponent_id,
            "opponent_name": OPPONENTS[self.opponent_id][0],
            "round": game.round + 1,
            "phase": game.phase,
            "wild": COLORS[game.wild] if not game.done else None,
            "current_player": game.current_player,
            "start_player": game.start_player,
            "next_start_player": game.next_start_player,
            "done": game.done,
            "winner": winner,
            "players": [_player_payload(player, i) for i, player in enumerate(game.players)],
            "factories": [dict(zip(COLORS, factory)) for factory in game.factories],
            "center_pool": dict(zip(COLORS, game.center_pool)),
            "supply": dict(zip(COLORS, game.supply)),
            "pending_bonus": game.pending_bonus,
            "last_ai_actions": self.last_ai_actions,
            "legal_actions": [_action_payload(game, action) for action in legal],
        }


SESSION = GameSession()


class AzulWebHandler(BaseHTTPRequestHandler):
    server_version = "AzulLocal/1.0"

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {self.address_string()} {format % args}")

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc: Exception, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": str(exc)}, status)

    def _read_json(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if size > 32_768:
            raise ValueError("Request is too large")
        return json.loads(self.rfile.read(size) or b"{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            with SESSION.lock:
                self._json(SESSION.payload())
            return
        static_files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        if path not in static_files:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = static_files[path]
        body = (STATIC_ROOT / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            data = self._read_json()
            if path == "/api/new":
                opponent = str(data.get("opponent", "competitive_hybrid"))
                seed = data.get("seed")
                self._json(SESSION.new_game(opponent, int(seed) if seed is not None else None))
                return
            if path == "/api/action":
                self._json(SESSION.act(int(data["action"])))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._error(exc, HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError) as exc:
            self._error(exc)
        except Exception as exc:
            self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), AzulWebHandler)
    print(f"Azul web game: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web UI for playing Azul against the AI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
