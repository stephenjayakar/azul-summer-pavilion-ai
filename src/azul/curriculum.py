from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .agents import HeuristicAgent, MultiStarHeuristicAgent, PolicyValueNet, ScoreHeuristicAgent, load_network
from .game import AzulGame
from .training import save_checkpoint


def collect_teacher_games(count: int, seed: int, teacher_kind: str = "balanced"):
    if teacher_kind not in {"balanced", "outer-score", "multi-outer"}:
        raise ValueError(f"unknown teacher: {teacher_kind}")
    if teacher_kind == "outer-score":
        teacher = ScoreHeuristicAgent(seed)
    elif teacher_kind == "multi-outer":
        teacher = MultiStarHeuristicAgent(seed)
    else:
        teacher = HeuristicAgent(seed)
    rows = []
    scores = []
    for i in range(count):
        game = AzulGame(seed + i)
        while not game.done:
            action = teacher.choose(game)
            rows.append((game.observation(), game.legal_mask(), action))
            game.step(action)
        scores.append((game.players[0].score, game.players[1].score))
        if (i + 1) % 20 == 0:
            print(json.dumps({"teacher_games": i + 1, "examples": len(rows)}), flush=True)
    return rows, np.asarray(scores)


def distill(
    base: str | None, output: str, games: int = 200, epochs: int = 12,
    seed: int = 9001, teacher_kind: str = "balanced",
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    begun = time.time()
    rows, scores = collect_teacher_games(games, seed, teacher_kind)
    if base:
        net = load_network(base, device)
    else:
        net = PolicyValueNet().to(device)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=4e-4, weight_decay=1e-4)
    obs = torch.from_numpy(np.stack([x[0] for x in rows]))
    masks = torch.from_numpy(np.stack([x[1] for x in rows]))
    targets = torch.tensor([x[2] for x in rows], dtype=torch.long)
    n = len(rows)
    for epoch in range(epochs):
        correct = total = 0
        losses = []
        for ids in torch.randperm(n).split(2048):
            batch_obs = obs[ids].to(device)
            batch_masks = masks[ids].to(device)
            target = targets[ids].to(device)
            logits, _ = net(batch_obs)
            logits = logits.masked_fill(~batch_masks, -1e9)
            # Standard label smoothing assigns probability mass to every class,
            # including masked illegal actions whose logits are -1e9. Use exact
            # legal teacher targets for this variable-legal-action policy.
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
            correct += int((logits.argmax(1) == target).sum())
            total += len(ids)
        print(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses)), "accuracy": correct / total}), flush=True)
    # Lower learning rate is retained when PPO resumes this checkpoint.
    for group in optimizer.param_groups:
        group["lr"] = 7e-5
    config = {
        "method": "heuristic curriculum distillation", "teacher": teacher_kind,
        "teacher_games": games,
        "examples": n, "epochs": epochs, "seed": seed,
        "teacher_mean_score": float(scores.mean()), "elapsed_s": round(time.time() - begun, 1),
    }
    path = Path(output)
    save_checkpoint(path, net, optimizer, 0, config)
    path.with_suffix(".json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    return path


def main():
    p = argparse.ArgumentParser(description="Distill the strategic curriculum into a neural policy")
    p.add_argument("--base")
    p.add_argument("--output", default="checkpoints/curriculum.pt")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--teacher", choices=("balanced", "outer-score", "multi-outer"), default="balanced")
    args = p.parse_args()
    distill(args.base, args.output, args.games, args.epochs, args.seed, args.teacher)


if __name__ == "__main__":
    main()
