"""Distill rollout policy improvement into a faster neural checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .agents import RolloutAgent, load_network
from .game import AzulGame
from .training import save_checkpoint


def collect_search_games(
    checkpoint: str, games_count: int, seed: int, device: torch.device,
    top_k: int, search_from_round: int, objective: str, individual_weight: float = 1.0,
):
    teacher = RolloutAgent(
        checkpoint, device=str(device), top_k=top_k,
        search_from_round=search_from_round, objective=objective,
        individual_weight=individual_weight,
    )
    games = [AzulGame(seed + i) for i in range(games_count)]
    rows = []
    while any(not game.done for game in games):
        active = [game for game in games if not game.done]
        observations = [game.observation() for game in active]
        masks = [game.legal_mask() for game in active]
        base_actions = teacher.neural.choose_many(active)
        actions = teacher.choose_many(active)
        rows.extend(
            (observation, mask, action, action != base_action)
            for observation, mask, action, base_action in zip(observations, masks, actions, base_actions)
        )
        for game, action in zip(active, actions):
            game.step(action)
    scores = np.asarray([(game.players[0].score, game.players[1].score) for game in games])
    return rows, scores


def distill_search(
    base: str, output: str, games: int = 64, epochs: int = 8,
    top_k: int = 3, search_from_round: int = 1, objective: str = "team",
    learning_rate: float = 5e-5, changed_weight: float = 20.0,
    individual_weight: float = 1.0, seed: int = 41_001,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    begun = time.time()
    rows, scores = collect_search_games(
        base, games, seed, device, top_k, search_from_round, objective, individual_weight
    )
    net = load_network(base, device); net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    obs = torch.from_numpy(np.stack([row[0] for row in rows]))
    masks = torch.from_numpy(np.stack([row[1] for row in rows]))
    targets = torch.tensor([row[2] for row in rows], dtype=torch.long)
    weights = torch.tensor([changed_weight if row[3] else 1.0 for row in rows], dtype=torch.float32)
    n = len(rows); history = []
    for epoch in range(epochs):
        correct = total = 0; losses = []
        for ids in torch.randperm(n).split(8192):
            batch_obs = obs[ids].to(device); batch_masks = masks[ids].to(device)
            target = targets[ids].to(device)
            weight = weights[ids].to(device)
            logits, _ = net(batch_obs); logits.masked_fill_(~batch_masks, -1e9)
            per_item = F.cross_entropy(logits, target, reduction="none")
            loss = (per_item * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            losses.append(float(loss)); correct += int((logits.argmax(1) == target).sum()); total += len(ids)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "accuracy": correct / total}
        history.append(row); print(json.dumps(row), flush=True)
    config = {
        "method": "rollout search distillation", "base": base, "games": games,
        "examples": n, "epochs": epochs, "top_k": top_k,
        "search_from_round": search_from_round, "objective": objective,
        "learning_rate": learning_rate, "changed_weight": changed_weight,
        "individual_weight": individual_weight,
        "changed_actions": sum(row[3] for row in rows),
        "changed_action_rate": sum(row[3] for row in rows) / n,
        "seed": seed, "device": str(device),
        "teacher_mean_score": float(scores.mean()),
        "teacher_mean_combined": float(scores.sum(axis=1).mean()),
        "teacher_max_score": int(scores.max()),
        "teacher_max_combined": int(scores.sum(axis=1).max()),
        "elapsed_s": round(time.time() - begun, 1), "history": history,
    }
    path = Path(output); save_checkpoint(path, net, optimizer, 0, config)
    path.with_suffix(".json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True); return path


def main():
    parser = argparse.ArgumentParser(description="Distill full-game rollout search into a neural policy")
    parser.add_argument("--base", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=64); parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--search-from-round", type=int, choices=range(6), default=1)
    parser.add_argument("--objective", choices=("own", "team", "frontier", "builder"), default="team")
    parser.add_argument("--individual-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--changed-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=41_001)
    args = parser.parse_args()
    distill_search(
        args.base, args.output, args.games, args.epochs, args.top_k,
        args.search_from_round, args.objective, args.learning_rate,
        args.changed_weight, args.individual_weight, args.seed,
    )


if __name__ == "__main__":
    main()
