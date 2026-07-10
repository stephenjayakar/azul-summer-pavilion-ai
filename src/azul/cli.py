from __future__ import annotations

import argparse
import json

from .agents import (
    HybridAgent, HeuristicAgent, NeuralAgent, RandomAgent, RoleAgent, RolloutAgent, evaluate,
    evaluate_neural_batched, evaluate_self_play_batched,
)
from .game import AzulGame, COLORS


def render(game: AzulGame) -> None:
    print(f"\nROUND {game.round + 1}/6 · {game.phase.upper()} · wild={COLORS[game.wild]}")
    print("Factories:", " | ".join(
        f"{i + 1}:" + ",".join(f"{COLORS[c][0]}{n}" for c, n in enumerate(f) if n)
        for i, f in enumerate(game.factories)
    ))
    print("Center:", ", ".join(f"{COLORS[c]}={n}" for c, n in enumerate(game.center_pool) if n) or "empty")
    print("Supply:", ", ".join(f"{COLORS[c]}={n}" for c, n in enumerate(game.supply) if n))
    for i, p in enumerate(game.players):
        inv = ",".join(f"{COLORS[c][0]}{n}" for c, n in enumerate(p.inventory) if n) or "empty"
        stars = " ".join(f"{COLORS[s][0].upper()}:{sum(star)}/6" for s, star in enumerate(p.outer))
        print(f"P{i} score={p.score} tiles=[{inv}] center={sum(c >= 0 for c in p.center)}/6 {stars}")


def play(checkpoint: str | None, difficulty: str, mode: str) -> None:
    if checkpoint:
        ai = HybridAgent(checkpoint) if mode == "hybrid" else NeuralAgent(checkpoint)
    elif difficulty == "random":
        ai = RandomAgent()
    else:
        ai = HeuristicAgent()
    game = AzulGame()
    print("You are P0. Enter the displayed choice number, or q to quit.")
    while not game.done:
        if game.current_player == 1:
            action = ai.choose(game)
            print(f"AI: {game.action_description(action)}")
            game.step(action)
            continue
        render(game)
        legal = game.legal_actions()
        for i, action in enumerate(legal):
            print(f"  {i + 1:2d}. {game.action_description(action)}")
        while True:
            raw = input("Your choice: ").strip().lower()
            if raw in {"q", "quit", "exit"}:
                return
            try:
                selected = int(raw) - 1
                if 0 <= selected < len(legal):
                    break
            except ValueError:
                pass
            print("Please enter one of the displayed numbers.")
        game.step(legal[selected])
    render(game)
    if game.players[0].score == game.players[1].score:
        print("Tie game.")
    else:
        print("You win!" if game.players[0].score > game.players[1].score else "AI wins.")


def main():
    parser = argparse.ArgumentParser(description="Azul: Summer Pavilion simulator")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play", help="play against an agent")
    p.add_argument("--checkpoint")
    p.add_argument("--difficulty", choices=("random", "heuristic"), default="heuristic")
    p.add_argument("--mode", choices=("hybrid", "neural"), default="hybrid",
                   help="with a checkpoint, hybrid is the strongest playable mode")
    e = sub.add_parser("evaluate", help="benchmark a checkpoint")
    e.add_argument("checkpoint")
    e.add_argument("--opponent", choices=("random", "heuristic"), default="random")
    e.add_argument("--games", type=int, default=200)
    e.add_argument("--mode", choices=("neural", "hybrid"), default="neural")
    s = sub.add_parser("score", help="fixed-seed neural self-play score benchmark")
    s.add_argument("checkpoint")
    s.add_argument("--games", type=int, default=500)
    s.add_argument("--stochastic", action="store_true")
    s.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--search-from-round", type=int, choices=range(0, 6))
    s.add_argument("--top-k", type=int, default=6)
    s.add_argument("--search-objective", choices=("own", "team", "frontier", "builder"), default="own")
    s.add_argument("--individual-weight", type=float, default=1.0)
    s.add_argument("--nested-search-from-round", type=int, choices=range(0, 6))
    s.add_argument("--nested-top-k", type=int, default=3)
    rs = sub.add_parser("role-score", help="fixed-seed builder/support checkpoint benchmark")
    rs.add_argument("checkpoint"); rs.add_argument("--games", type=int, default=500)
    args = parser.parse_args()
    if args.cmd == "play":
        play(args.checkpoint, args.difficulty, args.mode)
    elif args.cmd == "evaluate":
        opponent = RandomAgent(123) if args.opponent == "random" else HeuristicAgent(123)
        if args.mode == "hybrid":
            agent = HybridAgent(args.checkpoint)
            result = evaluate(agent, opponent, args.games)
        else:
            agent = NeuralAgent(args.checkpoint)
            result = evaluate_neural_batched(agent, opponent, args.games)
        print(json.dumps(result, indent=2))
    elif args.cmd == "score":
        if args.search_from_round is not None:
            if args.stochastic:
                parser.error("--stochastic and --search-from-round cannot be combined")
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            nested = None
            if args.nested_search_from_round is not None:
                nested = RolloutAgent(
                    args.checkpoint, device=device, top_k=args.nested_top_k,
                    search_from_round=args.nested_search_from_round,
                    objective=args.search_objective,
                    individual_weight=args.individual_weight,
                )
            agent = RolloutAgent(
                args.checkpoint, device=device,
                top_k=args.top_k, search_from_round=args.search_from_round,
                objective=args.search_objective,
                individual_weight=args.individual_weight,
                rollout_agent=nested,
            )
        else:
            agent = NeuralAgent(
                args.checkpoint, device=args.device, stochastic=args.stochastic,
                temperature=args.temperature,
            )
        print(json.dumps(evaluate_self_play_batched(agent, args.games), indent=2))
    else:
        agent = RoleAgent(args.checkpoint, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
        print(json.dumps(evaluate_self_play_batched(agent, args.games), indent=2))


if __name__ == "__main__":
    main()
