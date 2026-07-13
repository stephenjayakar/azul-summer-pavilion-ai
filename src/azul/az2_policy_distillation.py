"""Fit the AZ2 policy residual to search visits without changing values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .az2 import load_az2_checkpoint, save_az2_checkpoint
from .az2_training import ReplayData, load_replay, train_epoch


def policy_only_parameters(net: torch.nn.Module) -> list[torch.nn.Parameter]:
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in net.policy.parameters():
        parameter.requires_grad_(True)
    return [parameter for parameter in net.parameters() if parameter.requires_grad]


@torch.inference_mode()
def policy_metrics(
    net: torch.nn.Module, data: ReplayData, device: torch.device,
) -> dict[str, float]:
    net.eval(); losses = []; correct = 0; rows = 0
    target_entropy = []
    for start in range(0, len(data), 4096):
        sl = slice(start, start + 4096)
        obs = torch.from_numpy(data.obs[sl]).to(device)
        masks = torch.from_numpy(data.masks[sl]).to(device)
        targets = torch.from_numpy(data.policies[sl]).to(device)
        logits, _ = net(obs); logits.masked_fill_(~masks, -1e9)
        log_probs = F.log_softmax(logits, dim=1)
        losses.extend((-(targets * log_probs).sum(dim=1)).cpu().tolist())
        target_entropy.extend((-(targets * targets.clamp_min(1e-12).log()).sum(dim=1)).cpu().tolist())
        correct += int((logits.argmax(dim=1) == targets.argmax(dim=1)).sum())
        rows += len(obs)
    entropy = float(np.mean(target_entropy))
    cross_entropy = float(np.mean(losses))
    return {
        "policy_cross_entropy": cross_entropy,
        "policy_kl": cross_entropy - entropy,
        "policy_top1": correct / rows,
        "target_entropy": entropy,
        "examples": rows,
    }


def sharpen_targets(
    data: ReplayData, temperature: float, hard_weight: float,
) -> ReplayData:
    if temperature <= 0:
        raise ValueError("target_temperature must be positive")
    if not 0 <= hard_weight <= 1:
        raise ValueError("hard_weight must be between zero and one")
    policies = np.power(
        data.policies, 1.0 / temperature, where=data.policies > 0,
        out=np.zeros_like(data.policies),
    )
    policies /= policies.sum(axis=1, keepdims=True)
    if hard_weight:
        hard = np.zeros_like(policies)
        hard[np.arange(len(hard)), policies.argmax(axis=1)] = 1.0
        policies = (1 - hard_weight) * policies + hard_weight * hard
    return ReplayData(
        data.obs, data.masks, policies, data.values,
        data.value_valid, data.weights,
    )


def distill(
    base: str, replay_paths: list[str], output: str,
    epochs: int = 20, patience: int = 4, learning_rate: float = 3e-4,
    batch_size: int = 2048, holdout_fraction: float = 0.2,
    seed: int = 20260713, target_temperature: float = 1.0,
    hard_weight: float = 0.0,
) -> Path:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = sharpen_targets(
        ReplayData.concat([load_replay(path) for path in replay_paths]),
        target_temperature, hard_weight,
    )
    order = np.random.default_rng(seed).permutation(len(data))
    split = max(1, int(len(order) * (1 - holdout_fraction)))
    training = data.subset(order[:split]); holdout = data.subset(order[split:])
    net = load_az2_checkpoint(base, device)
    optimizer = torch.optim.AdamW(
        policy_only_parameters(net), lr=learning_rate, weight_decay=1e-4,
    )
    before = policy_metrics(net, holdout, device)
    best_loss = before["policy_cross_entropy"]
    best_state = {
        key: value.detach().cpu().clone() for key, value in net.state_dict().items()
    }
    stale = 0; history = []
    for epoch in range(1, epochs + 1):
        losses = train_epoch(
            net, optimizer, training, device, epochs=1, batch_size=batch_size,
            policy_weight=1.0, value_weight=0.0,
        )
        metrics = policy_metrics(net, holdout, device)
        row = {"epoch": epoch, **losses, **metrics}
        history.append(row); print(json.dumps(row), flush=True)
        if metrics["policy_cross_entropy"] < best_loss:
            best_loss = metrics["policy_cross_entropy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(best_state)
    report = {
        "method": "held-out search-policy distillation with frozen value paths",
        "base": base, "replay_paths": replay_paths, "device": str(device),
        "learning_rate": learning_rate, "holdout_fraction": holdout_fraction,
        "target_temperature": target_temperature, "hard_weight": hard_weight,
        "before": before, "selected": policy_metrics(net, holdout, device),
        "history": history,
    }
    return save_az2_checkpoint(output, net, None, -1, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill AZ2 search visits into policy")
    parser.add_argument("--base", required=True)
    parser.add_argument("--replay", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=0.0)
    args = parser.parse_args()
    path = distill(
        args.base, args.replay, args.output, args.epochs, args.patience,
        args.learning_rate, args.batch_size, args.holdout_fraction, args.seed,
        args.target_temperature, args.hard_weight,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
