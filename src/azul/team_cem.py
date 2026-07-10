"""Elite joint-trajectory search for cooperative high-score self-play."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .agents import PolicyValueNet
from .game import AzulGame
from .training import save_checkpoint


def collect_joint_population(
    net: PolicyValueNet, games_count: int, device: torch.device, seed: int,
    temperature: float, individual_weight: float,
):
    games = [AzulGame(seed + i) for i in range(games_count)]
    trajectories = [[] for _ in games]
    net.eval(); steps = 0
    while any(not game.done for game in games):
        active = [(i, game) for i, game in enumerate(games) if not game.done]
        obs_np = np.stack([game.observation() for _, game in active])
        masks_np = np.stack([game.legal_mask() for _, game in active])
        obs = torch.from_numpy(obs_np).to(device)
        masks = torch.from_numpy(masks_np).to(device)
        with torch.inference_mode():
            logits, _ = net(obs)
            logits = logits.masked_fill(~masks, -1e9) / temperature
            actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for row, (game_index, game) in enumerate(active):
            trajectories[game_index].append((obs_np[row], masks_np[row], int(actions[row])))
            game.step(int(actions[row])); steps += 1
    rows = []
    score_pairs = []
    for game, trajectory in zip(games, trajectories):
        pair = (game.players[0].score, game.players[1].score)
        combined = sum(pair)
        quality = combined + individual_weight * max(pair)
        rows.append((quality, combined, max(pair), trajectory))
        score_pairs.append(pair)
    return rows, np.asarray(score_pairs), steps


def joint_elite_update(
    net, optimizer, rows, quantile, device, epochs=4, batch_size=8192,
    entropy_coef=.004,
):
    qualities = np.asarray([row[0] for row in rows], dtype=np.float32)
    threshold = float(np.quantile(qualities, quantile))
    elite = [row for row in rows if row[0] >= threshold]
    records = [(quality, item) for quality, _, _, trajectory in elite for item in trajectory]
    obs = torch.from_numpy(np.stack([item[0] for _, item in records]))
    masks = torch.from_numpy(np.stack([item[1] for _, item in records]))
    actions = torch.tensor([item[2] for _, item in records], dtype=torch.long)
    weights = torch.tensor([
        1.0 + min(2.0, max(0.0, (quality - threshold) / 20.0))
        for quality, _ in records
    ], dtype=torch.float32)
    losses = []; accuracies = []; entropies = []; net.train(); n = len(records)
    for _ in range(epochs):
        for ids in torch.randperm(n).split(batch_size):
            batch_obs = obs[ids].to(device); batch_masks = masks[ids].to(device)
            target = actions[ids].to(device); weight = weights[ids].to(device)
            logits, _ = net(batch_obs); logits.masked_fill_(~batch_masks, -1e9)
            per_item = F.cross_entropy(logits, target, reduction="none")
            policy_loss = (per_item * weight).sum() / weight.sum()
            entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
            loss = policy_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            losses.append(float(policy_loss)); entropies.append(float(entropy))
            accuracies.append(float((logits.argmax(1) == target).float().mean()))
    return {
        "elite_threshold": threshold,
        "elite_games": len(elite),
        "elite_mean_combined": float(np.mean([row[1] for row in elite])),
        "elite_max_combined": int(max(row[1] for row in elite)),
        "elite_max_individual": int(max(row[2] for row in elite)),
        "elite_records": n,
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(accuracies)),
        "entropy": float(np.mean(entropies)),
    }


def train_team_cem(
    resume: str, output: str, iterations=6, games=1024, quantile=.9,
    temperature=1.5, individual_weight=.25, epochs=4, batch_size=8192,
    learning_rate=6e-5, entropy_coef=.004, seed=31_001,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    data = torch.load(resume, map_location=device, weights_only=False)
    net = PolicyValueNet(obs_size=data.get("obs_size", 282), hidden=data.get("hidden", 256)).to(device)
    net.load_state_dict(data["model"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    config = {
        "method": "elite joint-trajectory CEM", "resume": resume,
        "iterations": iterations, "games": games, "quantile": quantile,
        "temperature": temperature, "individual_weight": individual_weight,
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "entropy_coef": entropy_coef, "seed": seed, "device": str(device),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    begun = time.time()
    for iteration in range(iterations):
        rows, scores, steps = collect_joint_population(
            net, games, device, seed + iteration * 100_000, temperature, individual_weight
        )
        update = joint_elite_update(
            net, optimizer, rows, quantile, device, epochs, batch_size, entropy_coef
        )
        checkpoint = out / f"iteration_{iteration + 1:03d}.pt"
        save_checkpoint(checkpoint, net, optimizer, iteration, config)
        combined = scores.sum(axis=1)
        metrics = {
            "iteration": iteration + 1, "steps": steps,
            "mean_score": float(scores.mean()), "mean_combined": float(combined.mean()),
            "p90_combined": float(np.percentile(combined, 90)),
            "max_combined": int(combined.max()), "max_individual": int(scores.max()),
            **update, "elapsed_s": round(time.time() - begun, 1),
        }
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
    latest = out / "latest.pt"; save_checkpoint(latest, net, optimizer, iterations - 1, config)
    return latest


def main():
    parser = argparse.ArgumentParser(description="Elite cooperative score trajectory training")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output", default="checkpoints/team_cem")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--quantile", type=float, default=.9)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--individual-weight", type=float, default=.25)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--entropy-coef", type=float, default=.004)
    parser.add_argument("--seed", type=int, default=31_001)
    args = parser.parse_args()
    path = train_team_cem(
        args.resume, args.output, args.iterations, args.games, args.quantile,
        args.temperature, args.individual_weight, args.epochs, args.batch_size,
        args.learning_rate, args.entropy_coef, args.seed,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
