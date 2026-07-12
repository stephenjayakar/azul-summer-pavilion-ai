"""Persistent, replay-friendly logs for played games."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


LOG_SCHEMA_VERSION = 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class GameLog:
    """Append structured events to one JSONL file for a game session."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.game_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
        self.path = self.log_dir / f"{self.game_id}.jsonl"
        self.action_count = 0
        self.finished = False

    def _append(self, event: dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": LOG_SCHEMA_VERSION,
            "game_id": self.game_id,
            "timestamp": _timestamp(),
            **event,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")

    def start(self, *, seed: int | None, opponent_id: str, opponent_name: str, num_players: int) -> None:
        self._append({
            "event": "game_start",
            "seed": seed,
            "opponent_id": opponent_id,
            "opponent_name": opponent_name,
            "num_players": num_players,
        })

    def action(self, record: dict) -> None:
        if self.finished:
            raise RuntimeError("Cannot append an action to a finished game log")
        self._append({"event": "action", "sequence": self.action_count, **record})
        self.action_count += 1

    def finish(self, *, done: bool, reason: str, round_number: int, phase: str, scores: list[int], winner: int | None) -> None:
        if self.finished:
            return
        self._append({
            "event": "game_end",
            "sequence": self.action_count,
            "done": done,
            "reason": reason,
            "round": round_number,
            "phase": phase,
            "scores": scores,
            "winner": winner,
        })
        self.finished = True
