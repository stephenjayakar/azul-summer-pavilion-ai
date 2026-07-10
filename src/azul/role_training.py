"""Dual-policy cooperative self-play with fixed builder and support roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .agents import PolicyValueNet
from .game import AzulGame, OBS_SIZE
from .training import _masked_dist, ppo_update, score_potential


def collect_role_self_play(nets, games_count, device, seed, builder_weight=3.0, shaping_weight=1.0):
    games = [AzulGame(seed + i) for i in range(games_count)]
    trajectories = [[[], []] for _ in games]; steps = 0
    for net in nets: net.eval()
    while any(not game.done for game in games):
        for role in (0, 1):
            active = [(i, game) for i, game in enumerate(games) if not game.done and game.current_player == role]
            if not active: continue
            obs_np = np.stack([game.observation() for _, game in active])
            masks_np = np.stack([game.legal_mask() for _, game in active])
            obs = torch.from_numpy(obs_np).to(device); masks = torch.from_numpy(masks_np).to(device)
            with torch.inference_mode():
                dist, values = _masked_dist(nets[role], obs, masks)
                actions = dist.sample(); logps = dist.log_prob(actions)
            for row, (game_i, game) in enumerate(active):
                before_scores = [player.score for player in game.players]
                potential_before = score_potential(game.players[role]) if shaping_weight else 0.0
                record = {
                    "obs": obs_np[row], "mask": masks_np[row], "action": int(actions[row]),
                    "logp": float(logps[row]), "value": float(values[row]), "reward": 0.0,
                }
                game.step(int(actions[row])); after_scores = [player.score for player in game.players]
                before_utility = sum(before_scores) + builder_weight * before_scores[0]
                after_utility = sum(after_scores) + builder_weight * after_scores[0]
                record["score_delta"] = after_utility - before_utility
                record["reward"] = record["score_delta"] / 20.0
                if shaping_weight:
                    record["reward"] += shaping_weight * (
                        score_potential(game.players[role]) - potential_before
                    ) / 20.0
                trajectories[game_i][role].append(record); steps += 1
    score_rows = np.asarray([[p.score for p in game.players] for game in games])
    records = [[], []]
    for game, pair in zip(games, trajectories):
        total_delta = sum(p.score for p in game.players) - 10
        total_delta += builder_weight * (game.players[0].score - 5)
        for role, trajectory in enumerate(pair):
            captured = sum(record["score_delta"] for record in trajectory)
            trajectory[-1]["reward"] += (total_delta - captured) / 20.0
            if shaping_weight:
                trajectory[-1]["reward"] -= shaping_weight * score_potential(game.players[role]) / 20.0
            gae = 0.0; next_value = 0.0
            for index in reversed(range(len(trajectory))):
                delta = trajectory[index]["reward"] + next_value - trajectory[index]["value"]
                gae = delta + gae
                trajectory[index]["adv"] = gae
                trajectory[index]["return"] = gae + trajectory[index]["value"]
                next_value = trajectory[index]["value"]
            records[role].extend(trajectory)
    return records, score_rows, steps


def save_role_checkpoint(path, nets, optimizers, iteration, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "models": [net.state_dict() for net in nets],
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "iteration": iteration, "obs_size": OBS_SIZE, "hidden": 256, "config": config,
    }, path)


def train_roles(
    base, output, iterations=30, games=512, builder_weight=3.0,
    learning_rate=7e-5, entropy_coef=.01, ppo_epochs=6,
    batch_size=8192, seed=51_001,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(base, map_location=device, weights_only=False)
    nets = [PolicyValueNet(data.get("obs_size", OBS_SIZE), data.get("hidden", 256)).to(device) for _ in range(2)]
    for net in nets: net.load_state_dict(data["model"])
    optimizers = [torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4) for net in nets]
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    config = {
        "method": "dual-role cooperative PPO", "base": base, "iterations": iterations,
        "games": games, "builder_weight": builder_weight, "learning_rate": learning_rate,
        "entropy_coef": entropy_coef, "ppo_epochs": ppo_epochs, "batch_size": batch_size,
        "seed": seed, "device": str(device),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    begun = time.time()
    for iteration in range(iterations):
        records, scores, steps = collect_role_self_play(
            nets, games, device, seed + iteration * 100_000, builder_weight
        )
        updates = [
            ppo_update(nets[role], optimizers[role], records[role], device, ppo_epochs, batch_size, entropy_coef)
            for role in (0, 1)
        ]
        checkpoint = out / f"iteration_{iteration + 1:03d}.pt"
        save_role_checkpoint(checkpoint, nets, optimizers, iteration, config)
        combined = scores.sum(axis=1)
        metrics = {
            "iteration": iteration + 1, "steps": steps,
            "mean_score": float(scores.mean()), "mean_builder": float(scores[:, 0].mean()),
            "mean_support": float(scores[:, 1].mean()), "max_score": int(scores.max()),
            "mean_combined": float(combined.mean()), "max_combined": int(combined.max()),
            "builder_policy_loss": float(updates[0][0]), "support_policy_loss": float(updates[1][0]),
            "elapsed_s": round(time.time() - begun, 1),
        }
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
    latest = out / "latest.pt"; save_role_checkpoint(latest, nets, optimizers, iterations - 1, config)
    return latest


def main():
    parser = argparse.ArgumentParser(description="Train builder/support cooperative policies")
    parser.add_argument("--base", required=True); parser.add_argument("--output", default="checkpoints/roles")
    parser.add_argument("--iterations", type=int, default=30); parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--builder-weight", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--entropy-coef", type=float, default=.01)
    parser.add_argument("--ppo-epochs", type=int, default=6); parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=51_001); args = parser.parse_args()
    train_roles(
        args.base, args.output, args.iterations, args.games, args.builder_weight,
        args.learning_rate, args.entropy_coef, args.ppo_epochs, args.batch_size, args.seed,
    )


if __name__ == "__main__": main()
