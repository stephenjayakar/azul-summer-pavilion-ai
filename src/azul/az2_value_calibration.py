"""Held-out value calibration stage for the two-timescale AZ2 curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .az2 import load_az2_checkpoint, save_az2_checkpoint
from .az2_training import ReplayData, load_replay, train_epoch, value_metrics


def value_only_parameters(
    net: torch.nn.Module, residual_only: bool = False,
) -> list[torch.nn.Parameter]:
    """Freeze every policy-producing path and return value-head parameters."""
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    modules = (net.value_mlp,) if residual_only else (
        net.value, net.value_linear, net.value_mlp,
    )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in net.parameters() if parameter.requires_grad]


def calibrate(
    base: str, replay_dir: str, output: str, train_generations: int = 8,
    holdout_generations: int = 2, epochs: int = 30, patience: int = 5,
    learning_rate: float = 3e-4, batch_size: int = 4096,
    margin_weight: float = 2.0, win_weight: float = 0.25,
    seed: int = 20260712, residual_only: bool = False,
) -> Path:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(Path(replay_dir).glob("generation_*.npz"))
    required = train_generations + holdout_generations
    if len(paths) < required:
        raise ValueError(f"need {required} replay generations, found {len(paths)}")
    selected = paths[-required:]
    train_paths = selected[:train_generations]
    holdout_paths = selected[train_generations:]
    training = ReplayData.concat([load_replay(path) for path in train_paths])
    holdout = ReplayData.concat([load_replay(path) for path in holdout_paths])

    net = load_az2_checkpoint(base, device)
    # This is deliberately a value-only stage. Freezing the tokenizer/encoder
    # as well as the policy head guarantees that calibration cannot silently
    # replace the policy whose search strength established the replay data.
    trainable = value_only_parameters(net, residual_only=residual_only)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    head_weights = (1.0, 1.0, margin_weight, win_weight)
    before = value_metrics(net, holdout, device)
    best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    before_mse = before["value_head_mse"]
    best_weighted = (
        before_mse[0] + before_mse[1] + margin_weight * before_mse[2]
        + win_weight * before_mse[3]
    ) / sum(head_weights)
    stale = 0; history = []
    for epoch in range(1, epochs + 1):
        losses = train_epoch(
            net, optimizer, training, device, epochs=1, batch_size=batch_size,
            policy_weight=0.0, value_weight=1.0,
            value_head_weights=head_weights,
        )
        metrics = value_metrics(net, holdout, device)
        mse = metrics["value_head_mse"]
        weighted = (
            mse[0] + mse[1] + margin_weight * mse[2] + win_weight * mse[3]
        ) / sum(head_weights)
        row = {"epoch": epoch, **losses, **metrics, "weighted_value_mse": weighted}
        history.append(row); print(json.dumps(row), flush=True)
        if weighted < best_weighted:
            best_weighted = weighted
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(best_state)
    report = {
        "method": "held-out value calibration before deep Expert Iteration",
        "base": base, "device": str(device),
        "train_generations": [path.name for path in train_paths],
        "holdout_generations": [path.name for path in holdout_paths],
        "head_weights": head_weights, "before": before,
        "residual_only": residual_only,
        "selected": value_metrics(net, holdout, device),
        "best_weighted_value_mse": best_weighted, "history": history,
    }
    return save_az2_checkpoint(output, net, None, -1, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate AZ2 values on held-out replay")
    parser.add_argument("--base", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-generations", type=int, default=8)
    parser.add_argument("--holdout-generations", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--margin-weight", type=float, default=2.0)
    parser.add_argument("--win-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--residual-only", action="store_true")
    args = parser.parse_args()
    path = calibrate(
        args.base, args.replay_dir, args.output, args.train_generations,
        args.holdout_generations, args.epochs, args.patience,
        args.learning_rate, args.batch_size, args.margin_weight,
        args.win_weight, args.seed, args.residual_only,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
