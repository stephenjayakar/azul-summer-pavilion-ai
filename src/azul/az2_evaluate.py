"""Reproducible evaluation for structured policy+search checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .agents import HeuristicAgent
from .az2 import load_az2_checkpoint
from .az2_training import (
    RawAZ2Agent, SearchAZ2Agent, collect_teacher_examples, evaluate_agents,
    evaluate_score, human_metrics, load_human_examples, value_metrics,
)


def evaluate(
    checkpoint: str, baseline: str | None, games: int, score_games: int,
    simulations: int, heuristic_prior_weight: float,
    value_utility_weight: float, seed: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_az2_checkpoint(checkpoint, device)
    raw = RawAZ2Agent(net, device)
    searched = SearchAZ2Agent(
        net, device, simulations, heuristic_prior_weight, value_utility_weight,
    )
    result = {
        "checkpoint": checkpoint,
        "device": str(device),
        "config": {
            "games": games, "score_games": score_games,
            "simulations": simulations,
            "heuristic_prior_weight": heuristic_prior_weight,
            "value_utility_weight": value_utility_weight, "seed": seed,
        },
        "search_vs_heuristic": evaluate_agents(
            searched, HeuristicAgent(123), games, seed,
        ),
        "raw_score": evaluate_score(raw, score_games, seed + 100_000),
        "human": human_metrics(net, load_human_examples(), device),
    }
    validation = collect_teacher_examples(
        "checkpoints/best_search_policy.pt", 64, device, seed + 200_000,
        temperature=1.0, hard_weight=0.0,
    )
    result["value_validation"] = value_metrics(net, validation, device)
    if baseline:
        baseline_net = load_az2_checkpoint(baseline, device)
        baseline_search = SearchAZ2Agent(
            baseline_net, device, simulations, heuristic_prior_weight,
            value_utility_weight,
        )
        result["search_vs_baseline"] = evaluate_agents(
            searched, baseline_search, games, seed + 300_000,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an AZ2 policy+search checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--baseline")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--score-games", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--heuristic-prior-weight", type=float, default=0.5)
    parser.add_argument("--value-utility-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=881000)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint, args.baseline, args.games, args.score_games,
        args.simulations, args.heuristic_prior_weight,
        args.value_utility_weight, args.seed,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
