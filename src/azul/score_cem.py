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
from .training import _masked_dist, save_checkpoint


def collect_population(net, games_count: int, device: torch.device, seed: int, temperature: float):
    games = [AzulGame(seed + i) for i in range(games_count)]
    trajectories = [[[], []] for _ in games]
    net.eval()
    steps = 0
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
        for row, (game_i, game) in enumerate(active):
            actor = game.current_player
            trajectories[game_i][actor].append((obs_np[row], masks_np[row], int(actions[row])))
            game.step(int(actions[row]))
            steps += 1
    player_rows = []
    outer_stars = center_stars = 0
    for game, pair in zip(games, trajectories):
        for player, trajectory in enumerate(pair):
            score = game.players[player].score
            player_rows.append((score, trajectory))
            outer_stars += sum(all(star) for star in game.players[player].outer)
            center_stars += int(all(c >= 0 for c in game.players[player].center))
    scores = np.asarray([score for score, _ in player_rows], dtype=np.float32)
    return player_rows, scores, steps, outer_stars / len(player_rows), center_stars / len(player_rows)


def elite_update(net, optimizer, player_rows, quantile, device, epochs=4, batch_size=8192, entropy_coef=.002):
    scores = np.asarray([score for score, _ in player_rows], dtype=np.float32)
    threshold = float(np.quantile(scores, quantile))
    elite = [(score, trajectory) for score, trajectory in player_rows if score >= threshold]
    records = [(score, item) for score, trajectory in elite for item in trajectory]
    obs = torch.from_numpy(np.stack([item[0] for _, item in records]))
    masks = torch.from_numpy(np.stack([item[1] for _, item in records]))
    actions = torch.tensor([item[2] for _, item in records], dtype=torch.long)
    # Higher-scoring elites exert somewhat more pressure without allowing a
    # single lucky game to dominate the update.
    weights = torch.tensor([
        1.0 + min(2.0, max(0.0, (score - threshold) / 15.0)) for score, _ in records
    ], dtype=torch.float32)
    losses = []; accuracies = []; entropies = []
    n = len(records)
    net.train()
    for _ in range(epochs):
        for ids in torch.randperm(n).split(batch_size):
            batch_obs = obs[ids].to(device)
            batch_masks = masks[ids].to(device)
            target = actions[ids].to(device)
            weight = weights[ids].to(device)
            logits, _ = net(batch_obs)
            logits.masked_fill_(~batch_masks, -1e9)
            per_item = F.cross_entropy(logits, target, reduction="none")
            policy_loss = (per_item * weight).sum() / weight.sum()
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy().mean()
            loss = policy_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(float(policy_loss))
            accuracies.append(float((logits.argmax(1) == target).float().mean()))
            entropies.append(float(entropy))
    return {
        "elite_threshold": threshold,
        "elite_players": len(elite),
        "elite_mean_score": float(np.mean([score for score, _ in elite])),
        "elite_records": n,
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(accuracies)),
        "entropy": float(np.mean(entropies)),
    }


def train_cem(
    resume: str, output: str, iterations=8, games=512, quantile=.8,
    temperature=1.15, epochs=4, batch_size=8192, learning_rate=1e-4,
    entropy_coef=.002, seed=24_001,
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
        "method": "elite score self-play (CEM)", "resume": resume,
        "iterations": iterations, "games": games, "quantile": quantile,
        "temperature": temperature, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "entropy_coef": entropy_coef,
        "seed": seed, "device": str(device),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    begun = time.time()
    for iteration in range(iterations):
        rows, scores, steps, outer_rate, center_rate = collect_population(
            net, games, device, seed + iteration * 100_000, temperature
        )
        update = elite_update(
            net, optimizer, rows, quantile, device, epochs, batch_size, entropy_coef
        )
        checkpoint = out / f"iteration_{iteration + 1:03d}.pt"
        save_checkpoint(checkpoint, net, optimizer, iteration, config)
        metrics = {
            "iteration": iteration + 1, "steps": steps,
            "mean_score": float(scores.mean()), "median_score": float(np.median(scores)),
            "p90_score": float(np.percentile(scores, 90)), "max_score": int(scores.max()),
            "outer_stars_per_player": outer_rate, "center_stars_per_player": center_rate,
            **update, "elapsed_s": round(time.time() - begun, 1),
        }
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
    latest = out / "latest.pt"
    save_checkpoint(latest, net, optimizer, iterations - 1, config)
    return latest


def main():
    parser = argparse.ArgumentParser(description="Elite score-maximizing self-play")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output", default="checkpoints/score_cem")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--quantile", type=float, default=.8)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=.002)
    parser.add_argument("--seed", type=int, default=24_001)
    args = parser.parse_args()
    latest = train_cem(
        args.resume, args.output, args.iterations, args.games, args.quantile,
        args.temperature, args.epochs, args.batch_size, args.learning_rate,
        args.entropy_coef, args.seed,
    )
    print(f"saved {latest}")


if __name__ == "__main__":
    main()
