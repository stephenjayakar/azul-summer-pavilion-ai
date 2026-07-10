from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from .agents import HybridAgent, HeuristicAgent, NeuralAgent, RandomAgent
from .game import AzulGame, CENTER_SOURCE, COLORS, decode_action


def strategy_profile(agent, opponent, games: int = 100, seed: int = 70_000) -> dict:
    totals = Counter()
    color_stars = Counter()
    costs = Counter()
    scores = []
    for i in range(games):
        game = AzulGame(seed + i)
        seat = i % 2
        agents = [opponent, opponent]
        agents[seat] = agent
        while not game.done:
            actor = game.current_player
            action_id = agents[actor].choose(game)
            if actor == seat:
                action = decode_action(action_id)
                totals[f"action_{action.kind}"] += 1
                if action.kind == "draft" and action.source == CENTER_SOURCE and game.next_start_player is None:
                    totals["first_player_tokens"] += 1
                if action.kind.startswith("place"):
                    cost = action.slot + 1
                    costs[cost] += 1
                    totals["wild_tiles_spent"] += cost - action.natural
                if action.kind == "keep":
                    totals["tiles_carried"] += 1
            game.step(action_id)
        p = game.players[seat]
        scores.append(p.score)
        totals["board_tiles"] += p.board_tile_count()
        totals["center_stars"] += int(all(c >= 0 for c in p.center))
        for color, star in enumerate(p.outer):
            color_stars[COLORS[color]] += int(all(star))
        for slot in range(4):
            complete = sum(p.outer[s][slot] for s in range(6)) + (p.center[slot] >= 0) == 7
            totals[f"all_{slot + 1}s"] += int(complete)
        for kind in ("window", "statue", "pillar"):
            totals[f"claimed_{kind}"] += sum(k == kind for k, _ in p.claimed)
    placements = sum(costs.values())
    return {
        "games": games,
        "score_mean": round(float(np.mean(scores)), 3),
        "score_std": round(float(np.std(scores)), 3),
        "board_tiles_per_game": round(totals["board_tiles"] / games, 3),
        "completed_stars_per_game": {
            **{c: round(color_stars[c] / games, 3) for c in COLORS},
            "center": round(totals["center_stars"] / games, 3),
        },
        "number_sets_per_game": {str(n): round(totals[f"all_{n}s"] / games, 3) for n in range(1, 5)},
        "features_per_game": {
            k: round(totals[f"claimed_{k}"] / games, 3) for k in ("window", "statue", "pillar")
        },
        "placement_cost_distribution": {
            str(cost): round(costs[cost] / placements, 4) if placements else 0 for cost in range(1, 7)
        },
        "wild_tiles_spent_per_game": round(totals["wild_tiles_spent"] / games, 3),
        "tiles_carried_per_game": round(totals["tiles_carried"] / games, 3),
        "first_player_tokens_per_game": round(totals["first_player_tokens"] / games, 3),
    }


def main():
    p = argparse.ArgumentParser(description="Summarize strategy from completed simulator games")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--agent", choices=("hybrid", "neural", "heuristic"), default="hybrid")
    p.add_argument("--opponent", choices=("random", "heuristic", "neural"), default="heuristic")
    p.add_argument("--games", type=int, default=100)
    args = p.parse_args()
    if args.agent == "hybrid":
        agent = HybridAgent(args.checkpoint)
    elif args.agent == "neural":
        agent = NeuralAgent(args.checkpoint)
    else:
        agent = HeuristicAgent(1)
    if args.opponent == "random":
        opponent = RandomAgent(2)
    elif args.opponent == "neural":
        opponent = NeuralAgent(args.checkpoint)
    else:
        opponent = HeuristicAgent(2)
    print(json.dumps(strategy_profile(agent, opponent, args.games), indent=2))


if __name__ == "__main__":
    main()
