from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .agents import HeuristicAgent, load_network
from .curriculum import collect_teacher_games
from .game import AzulGame
from .training import save_checkpoint


def collect_adversarial(net, count: int, device: torch.device, seed: int, teacher_mix: float):
    """Collect teacher corrections on states visited against a strong opponent."""
    rng = random.Random(seed)
    teacher = HeuristicAgent(seed)
    games = [AzulGame(seed + i) for i in range(count)]
    neural_seats = [i % 2 for i in range(count)]
    rows = []
    net.eval()
    while any(not game.done for game in games):
        neural_rows = [
            i for i, game in enumerate(games)
            if not game.done and game.current_player == neural_seats[i]
        ]
        if neural_rows:
            obs_np = np.stack([games[i].observation() for i in neural_rows])
            masks_np = np.stack([games[i].legal_mask() for i in neural_rows])
            targets = [teacher.choose(games[i]) for i in neural_rows]
            with torch.inference_mode():
                obs = torch.from_numpy(obs_np).to(device)
                masks = torch.from_numpy(masks_np).to(device)
                logits, _ = net(obs)
                actions = logits.masked_fill(~masks, -1e9).argmax(1).cpu().tolist()
            for row_i, game_i in enumerate(neural_rows):
                rows.append((obs_np[row_i], masks_np[row_i], targets[row_i]))
                action = targets[row_i] if rng.random() < teacher_mix else actions[row_i]
                games[game_i].step(action)
        # Advance one strong-opponent decision per game. If a move leaves the
        # same opponent active, the outer loop handles it again.
        for i, game in enumerate(games):
            if not game.done and game.current_player != neural_seats[i]:
                game.step(teacher.choose(game))
    scores = np.asarray([
        (game.players[neural_seats[i]].score, game.players[1 - neural_seats[i]].score)
        for i, game in enumerate(games)
    ])
    return rows, scores


def dagger(
    base: str, output: str, games: int = 120, anchor_games: int = 40,
    epochs: int = 16, teacher_mix: float = .25, seed: int = 12_345,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_network(base, device)
    begun = time.time()
    rows, scores = collect_adversarial(net, games, device, seed, teacher_mix)
    anchors, anchor_scores = collect_teacher_games(anchor_games, seed + 1_000_000)
    rows.extend(anchors)
    print(json.dumps({
        "adversarial_games": games, "examples": len(rows),
        "pretrain_neural_score": float(scores[:, 0].mean()),
        "pretrain_teacher_score": float(scores[:, 1].mean()),
    }), flush=True)

    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    obs = torch.from_numpy(np.stack([x[0] for x in rows]))
    masks = torch.from_numpy(np.stack([x[1] for x in rows]))
    targets = torch.tensor([x[2] for x in rows], dtype=torch.long)
    n = len(rows)
    for epoch in range(epochs):
        losses = []; correct = 0
        for ids in torch.randperm(n).split(8192):
            batch_obs = obs[ids].to(device)
            batch_masks = masks[ids].to(device)
            target = targets[ids].to(device)
            logits, _ = net(batch_obs)
            logits.masked_fill_(~batch_masks, -1e9)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
            correct += int((logits.argmax(1) == target).sum())
        print(json.dumps({
            "epoch": epoch + 1, "loss": float(np.mean(losses)), "accuracy": correct / n,
        }), flush=True)
    for group in optimizer.param_groups:
        group["lr"] = 7e-5
    config = {
        "method": "DAgger adversarial curriculum", "base": base,
        "adversarial_games": games, "anchor_games": anchor_games,
        "examples": n, "epochs": epochs, "teacher_mix": teacher_mix,
        "seed": seed, "device": str(device), "elapsed_s": round(time.time() - begun, 1),
    }
    path = Path(output)
    save_checkpoint(path, net, optimizer, 0, config)
    path.with_suffix(".json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    return path


def main():
    p = argparse.ArgumentParser(description="Aggregate strong actions on neural-visited states")
    p.add_argument("--base", required=True)
    p.add_argument("--output", default="checkpoints/dagger.pt")
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--anchor-games", type=int, default=40)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--teacher-mix", type=float, default=.25)
    p.add_argument("--seed", type=int, default=12_345)
    args = p.parse_args()
    dagger(args.base, args.output, args.games, args.anchor_games, args.epochs, args.teacher_mix, args.seed)


if __name__ == "__main__":
    main()
