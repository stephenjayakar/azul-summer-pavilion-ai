"""Distill builder-oriented rollout search into separate seat-role networks."""

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
from .role_training import save_role_checkpoint


def distill_role_search(
    base, output, games=48, epochs=10, top_k=5, search_from_round=1,
    individual_weight=3.0, changed_weight=30.0, learning_rate=5e-5,
    seed=61_001,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = RolloutAgent(
        base, device=str(device), top_k=top_k, search_from_round=search_from_round,
        objective="builder", individual_weight=individual_weight,
    )
    states = [AzulGame(seed + i) for i in range(games)]; rows = [[], []]; begun = time.time()
    while any(not game.done for game in states):
        active = [game for game in states if not game.done]
        roles = [game.current_player for game in active]
        observations = [game.observation() for game in active]
        masks = [game.legal_mask() for game in active]
        base_actions = teacher.neural.choose_many(active); actions = teacher.choose_many(active)
        for role, observation, mask, action, base_action in zip(roles, observations, masks, actions, base_actions):
            rows[role].append((observation, mask, action, action != base_action))
        for game, action in zip(active, actions): game.step(action)
    scores = np.asarray([[p.score for p in game.players] for game in states])
    nets = [load_network(base, device) for _ in range(2)]
    optimizers = [torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4) for net in nets]
    histories = []
    for role in (0, 1):
        obs = torch.from_numpy(np.stack([row[0] for row in rows[role]]))
        masks = torch.from_numpy(np.stack([row[1] for row in rows[role]]))
        targets = torch.tensor([row[2] for row in rows[role]], dtype=torch.long)
        weights = torch.tensor([changed_weight if row[3] else 1 for row in rows[role]], dtype=torch.float32)
        history = []; nets[role].train(); n = len(rows[role])
        for epoch in range(epochs):
            losses = []; correct = 0
            for ids in torch.randperm(n).split(8192):
                logits, _ = nets[role](obs[ids].to(device)); batch_masks = masks[ids].to(device)
                logits.masked_fill_(~batch_masks, -1e9); target = targets[ids].to(device)
                weight = weights[ids].to(device)
                loss_items = F.cross_entropy(logits, target, reduction="none")
                loss = (loss_items * weight).sum() / weight.sum()
                optimizers[role].zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(nets[role].parameters(), 1); optimizers[role].step()
                losses.append(float(loss)); correct += int((logits.argmax(1) == target).sum())
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "accuracy": correct / n})
        histories.append(history)
    config = {
        "method": "builder rollout role distillation", "base": base, "games": games,
        "epochs": epochs, "top_k": top_k, "search_from_round": search_from_round,
        "individual_weight": individual_weight, "changed_weight": changed_weight,
        "learning_rate": learning_rate, "seed": seed,
        "teacher_mean_score": float(scores.mean()),
        "teacher_mean_builder": float(scores[:, 0].mean()),
        "teacher_mean_support": float(scores[:, 1].mean()),
        "teacher_max_score": int(scores.max()), "teacher_max_combined": int(scores.sum(1).max()),
        "examples": [len(rows[0]), len(rows[1])],
        "changed_actions": [sum(row[3] for row in rows[0]), sum(row[3] for row in rows[1])],
        "histories": histories, "elapsed_s": round(time.time() - begun, 1),
    }
    path = Path(output); save_role_checkpoint(path, nets, optimizers, 0, config)
    path.with_suffix(".json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True); return path


def main():
    parser = argparse.ArgumentParser(description="Distill builder search into role policies")
    parser.add_argument("--base", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=48); parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5); parser.add_argument("--search-from-round", type=int, default=1)
    parser.add_argument("--individual-weight", type=float, default=3); parser.add_argument("--changed-weight", type=float, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-5); parser.add_argument("--seed", type=int, default=61_001)
    args = parser.parse_args(); distill_role_search(
        args.base, args.output, args.games, args.epochs, args.top_k,
        args.search_from_round, args.individual_weight, args.changed_weight,
        args.learning_rate, args.seed,
    )


if __name__ == "__main__": main()
