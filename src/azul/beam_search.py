"""Joint cooperative beam search over complete deterministic games."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .agents import NeuralAgent
from .game import AzulGame
from .training import score_potential


@dataclass
class BeamNode:
    game: AzulGame
    log_probability: float
    actions: tuple[int, ...]


def state_priority(node: BeamNode, policy_weight: float = .03) -> float:
    game = node.game
    if game.done:
        return sum(player.score for player in game.players) + policy_weight * node.log_probability
    score = sum(player.score for player in game.players)
    potential = sum(score_potential(player) for player in game.players)
    inventory = sum(sum(player.inventory) + sum(player.stored) for player in game.players)
    # Inventory is future construction capacity; potential estimates unfinished
    # end bonuses. A small policy prior stabilizes ties during draft-only steps.
    return score + potential + .35 * inventory + policy_weight * node.log_probability


@torch.inference_mode()
def plan_game(
    agent: NeuralAgent, seed: int, beam_width: int = 32, branch_width: int = 6,
    policy_weight: float = .03,
) -> BeamNode:
    beam = [BeamNode(AzulGame(seed), 0.0, ())]
    for _ in range(600):
        if all(node.game.done for node in beam):
            break
        live = [node for node in beam if not node.game.done]
        obs = torch.from_numpy(np.stack([node.game.observation() for node in live])).to(agent.device)
        masks = torch.from_numpy(np.stack([node.game.legal_mask() for node in live])).to(agent.device)
        logits, _ = agent.net(obs); logits.masked_fill_(~masks, -1e9)
        log_probs = torch.log_softmax(logits, dim=1)
        expanded = [node for node in beam if node.game.done]
        for row, node in enumerate(live):
            count = min(branch_width, len(node.game.legal_actions()))
            actions = torch.topk(logits[row], count).indices.tolist()
            for action in actions:
                clone = node.game.clone(); clone.step(action)
                expanded.append(BeamNode(
                    clone,
                    node.log_probability + float(log_probs[row, action]),
                    node.actions + (action,),
                ))
        beam = sorted(
            expanded, key=lambda node: state_priority(node, policy_weight), reverse=True
        )[:beam_width]
    else:
        raise RuntimeError("beam search exceeded 600 action layers")
    completed = [node for node in beam if node.game.done]
    if not completed:
        raise RuntimeError("beam search produced no complete game")
    return max(completed, key=lambda node: sum(player.score for player in node.game.players))


def beam_report(
    checkpoint: str, games: int = 20, seed: int = 50_000,
    beam_width: int = 32, branch_width: int = 6, policy_weight: float = .03,
    device: str = "cuda",
) -> dict:
    agent = NeuralAgent(checkpoint, device=device)
    pairs = []
    for game_index in range(games):
        result = plan_game(agent, seed + game_index, beam_width, branch_width, policy_weight)
        pair = tuple(player.score for player in result.game.players)
        pairs.append(pair)
        print(json.dumps({
            "game": game_index + 1, "scores": pair, "combined": sum(pair),
            "actions": len(result.actions),
        }), flush=True)
    scores = np.asarray(pairs)
    combined = scores.sum(axis=1)
    return {
        "checkpoint": checkpoint, "games": games, "seed": seed,
        "beam_width": beam_width, "branch_width": branch_width,
        "policy_weight": policy_weight,
        "mean_score": float(scores.mean()), "median_score": float(np.median(scores)),
        "max_individual_score": int(scores.max()),
        "mean_combined_score": float(combined.mean()),
        "max_combined_score": int(combined.max()),
        "score_pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description="Joint full-game cooperative beam planning")
    parser.add_argument("checkpoint"); parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--branch-width", type=int, default=6)
    parser.add_argument("--policy-weight", type=float, default=.03)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = beam_report(
        args.checkpoint, args.games, args.seed, args.beam_width,
        args.branch_width, args.policy_weight, args.device,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "score_pairs"}, indent=2))


if __name__ == "__main__":
    main()
