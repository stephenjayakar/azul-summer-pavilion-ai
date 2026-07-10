"""Observed two-player score-pair Pareto frontier from legal simulator games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import NeuralAgent
from .game import AzulGame


def nondominated_score_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return unique pairs not weakly dominated in both player scores."""
    ordered = sorted(set(pairs), key=lambda pair: (-pair[0], -pair[1]))
    frontier: list[tuple[int, int]] = []
    best_second = -1
    for pair in ordered:
        if pair[1] > best_second:
            frontier.append(pair)
            best_second = pair[1]
    return sorted(frontier)


def collect_pairs(
    checkpoint: str | Path, games: int, seed: int, *,
    stochastic: bool = False, device: str = "cpu",
) -> list[tuple[int, int]]:
    agent = NeuralAgent(checkpoint, device=device, stochastic=stochastic)
    states = [AzulGame(seed + i) for i in range(games)]
    while any(not game.done for game in states):
        active = [game for game in states if not game.done]
        actions = agent.choose_many(active)
        for game, action in zip(active, actions):
            game.step(action)
    return [(game.players[0].score, game.players[1].score) for game in states]


def empirical_report(
    checkpoints: list[str], games: int = 1000, seed: int = 120_000,
    include_stochastic: bool = True, device: str = "cpu",
) -> dict:
    records = []
    all_pairs: list[tuple[int, int]] = []
    modes = (False, True) if include_stochastic else (False,)
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for stochastic in modes:
            run_seed = seed + checkpoint_index * 100_000 + int(stochastic) * 50_000
            pairs = collect_pairs(checkpoint, games, run_seed, stochastic=stochastic, device=device)
            # Add the mirrored orientation: the frontier describes unlabeled
            # player score tradeoffs rather than first-seat advantage.
            all_pairs.extend(pairs)
            all_pairs.extend((b, a) for a, b in pairs)
            records.append({
                "checkpoint": checkpoint,
                "mode": "stochastic" if stochastic else "deterministic",
                "games": games,
                "seed": run_seed,
                "mean_p0": sum(a for a, _ in pairs) / games,
                "mean_p1": sum(b for _, b in pairs) / games,
                "max_individual": max(max(pair) for pair in pairs),
                "max_combined": max(sum(pair) for pair in pairs),
            })
    frontier = nondominated_score_pairs(all_pairs)
    max_total_pair = max(all_pairs, key=lambda pair: (sum(pair), min(pair)))
    max_floor_pair = max(all_pairs, key=lambda pair: (min(pair), sum(pair)))
    return {
        "interpretation": "Observed lower bound from fully legal games; more search can only expand this frontier.",
        "runs": records,
        "sampled_orientations": len(all_pairs),
        "max_individual_score": max(max(pair) for pair in all_pairs),
        "max_combined": {"scores": max_total_pair, "total": sum(max_total_pair)},
        "max_minimum_player_score": {"scores": max_floor_pair, "minimum": min(max_floor_pair)},
        "pareto_frontier": [{"p0_score": a, "p1_score": b, "total": a + b} for a, b in frontier],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a legal-game score-pair Pareto frontier")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--games", type=int, default=1000, help="games per checkpoint and mode")
    parser.add_argument("--seed", type=int, default=120_000)
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("empirical_frontier.json"))
    args = parser.parse_args()
    report = empirical_report(
        args.checkpoint, args.games, args.seed,
        include_stochastic=not args.deterministic_only, device=args.device,
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "sampled_orientations": report["sampled_orientations"],
        "frontier_points": len(report["pareto_frontier"]),
        "max_individual_score": report["max_individual_score"],
        "max_combined": report["max_combined"],
        "max_minimum_player_score": report["max_minimum_player_score"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
