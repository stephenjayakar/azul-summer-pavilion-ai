"""Persistent league/self-play training for the second-generation Azul agent."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from .agents import HeuristicAgent, NeuralAgent, load_network
from .az2 import (
    FastBatchedPUCT, FastNode, StructuredPolicyValueNet,
    load_az2_checkpoint, save_az2_checkpoint, value_targets,
)
from .game import AzulGame, NUM_ACTIONS, OBS_SIZE


@dataclass
class ReplayData:
    obs: np.ndarray
    masks: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    value_valid: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return len(self.obs)

    def subset(self, ids: np.ndarray) -> "ReplayData":
        return ReplayData(
            self.obs[ids], self.masks[ids], self.policies[ids],
            self.values[ids], self.value_valid[ids], self.weights[ids],
        )

    @staticmethod
    def concat(parts: list["ReplayData"]) -> "ReplayData":
        parts = [part for part in parts if len(part)]
        if not parts:
            return empty_replay()
        return ReplayData(*(
            np.concatenate([getattr(part, name) for part in parts], axis=0)
            for name in ("obs", "masks", "policies", "values", "value_valid", "weights")
        ))


def empty_replay() -> ReplayData:
    return ReplayData(
        np.empty((0, OBS_SIZE), np.float32), np.empty((0, NUM_ACTIONS), np.bool_),
        np.empty((0, NUM_ACTIONS), np.float32), np.empty((0, 4), np.float32),
        np.empty((0,), np.bool_), np.empty((0,), np.float32),
    )


def records_to_replay(records: list[dict]) -> ReplayData:
    if not records:
        return empty_replay()
    return ReplayData(
        np.stack([row["obs"] for row in records]).astype(np.float32),
        np.stack([row["mask"] for row in records]).astype(np.bool_),
        np.stack([row["policy"] for row in records]).astype(np.float32),
        np.stack([row.get("values", np.zeros(4, np.float32)) for row in records]).astype(np.float32),
        np.asarray([row.get("value_valid", True) for row in records], dtype=np.bool_),
        np.asarray([row.get("weight", 1.0) for row in records], dtype=np.float32),
    )


def save_replay(path: str | Path, data: ReplayData) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, obs=data.obs, masks=np.packbits(data.masks, axis=1),
        policies=data.policies.astype(np.float16), values=data.values,
        value_valid=data.value_valid, weights=data.weights,
    )
    return path


def load_replay(path: str | Path) -> ReplayData:
    with np.load(path) as data:
        masks = np.unpackbits(data["masks"], axis=1, count=NUM_ACTIONS).astype(np.bool_)
        return ReplayData(
            data["obs"].astype(np.float32), masks,
            data["policies"].astype(np.float32), data["values"].astype(np.float32),
            data["value_valid"].astype(np.bool_), data["weights"].astype(np.float32),
        )


def load_replay_window(
    directory: str | Path, generations: int, max_examples: int, seed: int,
) -> ReplayData:
    paths = sorted(Path(directory).glob("generation_*.npz"))[-generations:]
    combined = ReplayData.concat([load_replay(path) for path in paths])
    if max_examples and len(combined) > max_examples:
        ids = np.random.default_rng(seed).choice(len(combined), max_examples, replace=False)
        combined = combined.subset(ids)
    return combined


def load_human_examples(
    manifest_path: str | Path = "human_logs_manifest.json", include_limited: bool = False,
) -> ReplayData:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent / manifest["raw_directory"]
    complete = {row["file"]: row for row in manifest["complete_outcome"]}
    partial = {row["file"] for row in manifest["strong_partial"]}
    limited = set(manifest["limited_opening"]) if include_limited else set()
    selected = set(complete) | partial | limited
    records = []
    for name in sorted(selected):
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"human log missing: {path}")
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        actions = [event for event in events if event.get("event") == "action"]
        sequences = [event["sequence"] for event in actions]
        if sequences != list(range(len(actions))):
            raise ValueError(f"non-contiguous action sequence: {name}")
        for event in actions:
            if event.get("actor") != "human":
                continue
            action = int(event["action_id"])
            legal = [int(x) for x in event["legal_action_ids"]]
            obs = np.asarray(event["observation"], dtype=np.float32)
            if obs.shape != (OBS_SIZE,) or action not in legal:
                raise ValueError(f"malformed or illegal human row in {name}")
            mask = np.zeros(NUM_ACTIONS, dtype=np.bool_); mask[legal] = True
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32); policy[action] = 1.0
            is_complete = name in complete
            values = value_targets(complete[name]["scores"], int(event["player"])) if is_complete else np.zeros(4, np.float32)
            weight = 0.25 if is_complete else (0.15 if name in partial else 0.05)
            records.append({
                "obs": obs, "mask": mask, "policy": policy, "values": values,
                "value_valid": is_complete, "weight": weight,
            })
    return records_to_replay(records)


@torch.inference_mode()
def collect_teacher_examples(
    checkpoint: str, games_count: int, device: torch.device, seed: int,
    temperature: float = 0.8, hard_weight: float = 0.9,
) -> ReplayData:
    teacher = load_network(checkpoint, device)
    games = [AzulGame(seed + i) for i in range(games_count)]
    records: list[dict] = []
    while any(not game.done for game in games):
        active = [(i, game) for i, game in enumerate(games) if not game.done]
        obs_np = np.stack([game.observation() for _, game in active])
        masks_np = np.stack([game.legal_mask() for _, game in active])
        obs = torch.from_numpy(obs_np).to(device)
        masks = torch.from_numpy(masks_np).to(device)
        logits, _ = teacher(obs); logits.masked_fill_(~masks, -1e9)
        policies = torch.softmax(logits / temperature, dim=1).cpu().numpy()
        actions = logits.argmax(dim=1).cpu().tolist()
        policies *= 1.0 - hard_weight
        policies[np.arange(len(actions)), actions] += hard_weight
        for row, ((game_i, game), action) in enumerate(zip(active, actions)):
            records.append({
                "obs": obs_np[row], "mask": masks_np[row], "policy": policies[row],
                "game": game_i, "player": game.current_player,
            })
            game.step_fast(action)
    scores = [[player.score for player in game.players] for game in games]
    for row in records:
        row["values"] = value_targets(scores[row["game"]], row["player"])
        row["value_valid"] = True; row["weight"] = 1.0
    return records_to_replay(records)


@torch.inference_mode()
def collect_dagger_examples(
    student: StructuredPolicyValueNet, checkpoint: str, games_count: int,
    device: torch.device, seed: int, teacher_mix: float = 0.5,
    temperature: float = 0.8, hard_weight: float = 0.9,
) -> ReplayData:
    """Label student-visited states with the verified legacy policy."""
    teacher = load_network(checkpoint, device)
    games = [AzulGame(seed + i) for i in range(games_count)]
    rng = np.random.default_rng(seed); records = []
    student.eval()
    while any(not game.done for game in games):
        active = [(i, game) for i, game in enumerate(games) if not game.done]
        obs_np = np.stack([game.observation() for _, game in active])
        masks_np = np.stack([game.legal_mask() for _, game in active])
        obs = torch.from_numpy(obs_np).to(device)
        masks = torch.from_numpy(masks_np).to(device)
        teacher_logits, _ = teacher(obs); teacher_logits.masked_fill_(~masks, -1e9)
        student_logits, _ = student(obs); student_logits.masked_fill_(~masks, -1e9)
        policies = torch.softmax(teacher_logits / temperature, dim=1).cpu().numpy()
        teacher_actions = teacher_logits.argmax(dim=1).cpu().numpy()
        student_actions = student_logits.argmax(dim=1).cpu().numpy()
        policies *= 1 - hard_weight
        policies[np.arange(len(active)), teacher_actions] += hard_weight
        execute_teacher = rng.random(len(active)) < teacher_mix
        actions = np.where(execute_teacher, teacher_actions, student_actions)
        for row, ((game_i, game), action) in enumerate(zip(active, actions)):
            records.append({
                "obs": obs_np[row], "mask": masks_np[row], "policy": policies[row],
                "game": game_i, "player": game.current_player, "weight": 1.0,
            })
            game.step_fast(int(action))
    scores = [[player.score for player in game.players] for game in games]
    for row in records:
        row["values"] = value_targets(scores[row["game"]], row["player"])
        row["value_valid"] = True
    return records_to_replay(records)


def train_epoch(
    net: StructuredPolicyValueNet, optimizer: torch.optim.Optimizer,
    data: ReplayData, device: torch.device, epochs: int = 1,
    batch_size: int = 2048, value_weight: float = 1.0,
    entropy_coef: float = 0.001, policy_weight: float = 1.0,
    value_head_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> dict[str, float]:
    if not len(data):
        raise ValueError("empty training data")
    net.train(); metrics = []
    for _ in range(epochs):
        for ids_np in np.array_split(np.random.permutation(len(data)), max(1, int(np.ceil(len(data) / batch_size)))):
            ids = torch.from_numpy(ids_np)
            obs = torch.from_numpy(data.obs[ids_np]).to(device)
            masks = torch.from_numpy(data.masks[ids_np]).to(device)
            targets = torch.from_numpy(data.policies[ids_np]).to(device)
            value_targets_t = torch.from_numpy(data.values[ids_np]).to(device)
            valid = torch.from_numpy(data.value_valid[ids_np]).to(device)
            weights = torch.from_numpy(data.weights[ids_np]).to(device)
            logits, predicted = net(obs); logits.masked_fill_(~masks, -1e9)
            if not torch.isfinite(predicted).all() or not torch.isfinite(logits[masks]).all():
                raise FloatingPointError("non-finite policy/value prediction")
            log_probs = F.log_softmax(logits, dim=1); probs = log_probs.exp()
            per_policy = -(targets * log_probs).sum(dim=1)
            policy_loss = (per_policy * weights).sum() / weights.sum().clamp_min(1e-6)
            head_weights = torch.as_tensor(value_head_weights, device=device)
            per_value = (
                (predicted - value_targets_t).pow(2) * head_weights
            ).sum(dim=1) / head_weights.sum().clamp_min(1e-6)
            value_weights = weights * valid.float()
            value_loss = (per_value * value_weights).sum() / value_weights.sum().clamp_min(1e-6)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            loss = policy_weight * policy_loss + value_weight * value_loss - policy_weight * entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            metrics.append((float(policy_loss), float(value_loss), float(entropy)))
    mean = np.mean(metrics, axis=0)
    return {"policy_loss": float(mean[0]), "value_loss": float(mean[1]), "entropy": float(mean[2])}


@torch.inference_mode()
def human_metrics(net: StructuredPolicyValueNet, data: ReplayData, device: torch.device) -> dict[str, float]:
    net.eval(); losses = []; correct = 0; total = 0; value_errors = []
    for start in range(0, len(data), 2048):
        sl = slice(start, start + 2048)
        obs = torch.from_numpy(data.obs[sl]).to(device)
        masks = torch.from_numpy(data.masks[sl]).to(device)
        targets = torch.from_numpy(data.policies[sl]).to(device)
        logits, values = net(obs); logits.masked_fill_(~masks, -1e9)
        losses.extend((-(targets * F.log_softmax(logits, dim=1)).sum(dim=1)).cpu().tolist())
        correct += int((logits.argmax(dim=1) == targets.argmax(dim=1)).sum())
        total += len(obs)
        valid = data.value_valid[sl]
        if valid.any():
            value_errors.extend(np.abs(values.cpu().numpy()[valid] - data.values[sl][valid]).mean(axis=1).tolist())
    return {
        "human_policy_nll": float(np.mean(losses)),
        "human_top1": correct / total,
        "human_value_mae": float(np.mean(value_errors)) if value_errors else 0.0,
        "human_examples": total,
    }


@torch.inference_mode()
def value_metrics(
    net: StructuredPolicyValueNet, data: ReplayData, device: torch.device,
) -> dict[str, float | list[float]]:
    """Report held-out errors for every value head and their aggregate."""
    net.eval(); predictions = []
    for start in range(0, len(data), 4096):
        obs = torch.from_numpy(data.obs[start:start + 4096]).to(device)
        _, values = net(obs)
        predictions.append(values.cpu().numpy())
    predicted = np.concatenate(predictions)
    valid = data.value_valid
    absolute = np.abs(predicted[valid] - data.values[valid])
    squared = (predicted[valid] - data.values[valid]) ** 2
    return {
        "value_mae": float(absolute.mean()),
        "value_mse": float(squared.mean()),
        "value_head_mae": absolute.mean(axis=0).tolist(),
        "value_head_mse": squared.mean(axis=0).tolist(),
        "value_examples": int(valid.sum()),
    }


def _sample(policy: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 1e-6:
        return int(np.argmax(policy))
    scaled = np.power(policy, 1 / temperature, where=policy > 0, out=np.zeros_like(policy))
    scaled /= scaled.sum()
    return int(rng.choice(len(policy), p=scaled))


def collect_search_self_play(
    net: StructuredPolicyValueNet, games_count: int, device: torch.device,
    seed: int, simulations: int = 128, temperature_moves: int = 24,
    heuristic_prior_weight: float = 0.0,
    value_utility_weight: float = 1.0,
    dirichlet_fraction: float = 0.25,
    progress: Callable[[dict[str, int | float | str]], None] | None = None,
) -> tuple[ReplayData, dict[str, float]]:
    games = [AzulGame(seed + i) for i in range(games_count)]
    roots: list[FastNode | None] = [None] * games_count
    moves = [0] * games_count; records = []; rng = np.random.default_rng(seed)
    search = FastBatchedPUCT(
        net, device, simulations=simulations, seed=seed,
        heuristic_prior_weight=heuristic_prior_weight,
        value_utility_weight=value_utility_weight,
        dirichlet_fraction=dirichlet_fraction,
    )
    begun = time.time(); last_progress = 0.0
    while any(not game.done for game in games):
        active_ids = [i for i, game in enumerate(games) if not game.done]
        active_games = [games[i] for i in active_ids]
        policies, searched_roots = search.search_many(
            active_games, add_noise=True, roots=[roots[i] for i in active_ids],
        )
        actions = []
        for local, game_i in enumerate(active_ids):
            game = games[game_i]
            records.append({
                "obs": game.observation(), "mask": game.legal_mask(),
                "policy": policies[local], "game": game_i,
                "player": game.current_player, "weight": 1.0,
            })
            action = _sample(policies[local], 1.0 if moves[game_i] < temperature_moves else 0.0, rng)
            actions.append(action); game.step_fast(action); moves[game_i] += 1
        advanced = search.advance_roots(searched_roots, actions, active_games)
        for game_i, root in zip(active_ids, advanced):
            roots[game_i] = root
        now = time.time()
        if progress is not None and (now - last_progress >= 2.0 or all(game.done for game in games)):
            progress({
                "phase": "self_play", "completed_games": sum(game.done for game in games),
                "games": games_count, "moves": sum(moves), "elapsed_s": round(now - begun, 1),
            })
            last_progress = now
    scores = np.asarray([[player.score for player in game.players] for game in games])
    for game in games:
        game.assert_invariants()
    for row in records:
        row["values"] = value_targets(scores[row["game"]], row["player"])
        row["value_valid"] = True
    elapsed = time.time() - begun
    return records_to_replay(records), {
        "selfplay_examples": len(records), "selfplay_seconds": elapsed,
        "search_examples_per_second": len(records) / elapsed,
        "mean_score": float(scores.mean()),
        "mean_combined_score": float(scores.sum(axis=1).mean()),
        "mean_abs_margin": float(np.abs(scores[:, 0] - scores[:, 1]).mean()),
        "chance_outcomes": len(search.chance_outcomes_seen),
    }


def collect_search_vs_opponent(
    net: StructuredPolicyValueNet, opponent, games_count: int,
    device: torch.device, seed: int, simulations: int = 32,
    heuristic_prior_weight: float = 0.5,
    value_utility_weight: float = 0.25,
    temperature_moves: int = 12,
    progress: Callable[[dict[str, int | float | str]], None] | None = None,
) -> tuple[ReplayData, dict[str, float]]:
    """Collect search targets on the state distribution induced by an opponent."""
    games = [AzulGame(seed + i) for i in range(games_count)]
    agent_seats = [i % 2 for i in range(games_count)]
    pending: dict[int, tuple[FastNode, int]] = {}
    moves = [0] * games_count; records = []; rng = np.random.default_rng(seed)
    search = FastBatchedPUCT(
        net, device, simulations=simulations, seed=seed,
        heuristic_prior_weight=heuristic_prior_weight,
        value_utility_weight=value_utility_weight, dirichlet_fraction=0.0,
    )
    begun = time.time(); last_progress = 0.0
    while any(not game.done for game in games):
        own_ids = [
            i for i, game in enumerate(games)
            if not game.done and game.current_player == agent_seats[i]
        ]
        if own_ids:
            own_games = [games[i] for i in own_ids]
            reusable = []
            for game_i, game in zip(own_ids, own_games):
                prior = pending.pop(game_i, None)
                reusable.append(
                    search.recover_root_after_observed_moves(*prior, game)
                    if prior is not None else None
                )
            policies, roots = search.search_many(
                own_games, add_noise=False, roots=reusable,
            )
            for game_i, game, policy, root in zip(
                own_ids, own_games, policies, roots,
            ):
                records.append({
                    "obs": game.observation(), "mask": game.legal_mask(),
                    "policy": policy, "game": game_i,
                    "player": game.current_player, "weight": 1.0,
                })
                action = _sample(
                    policy, 1.0 if moves[game_i] < temperature_moves else 0.0,
                    rng,
                )
                game.step_fast(action); moves[game_i] += 1
                pending[game_i] = (root, action)
        other_ids = [
            i for i, game in enumerate(games)
            if not game.done and game.current_player != agent_seats[i]
        ]
        if hasattr(opponent, "choose_many"):
            actions = opponent.choose_many([games[i] for i in other_ids])
            for game_i, action in zip(other_ids, actions):
                games[game_i].step_fast(action)
        else:
            for game_i in other_ids:
                games[game_i].step_fast(opponent.choose(games[game_i]))
        now = time.time()
        if progress is not None and now - last_progress >= 2.0:
            progress({
                "phase": "opponent_search", "completed_games": sum(g.done for g in games),
                "games": games_count, "examples": len(records),
                "elapsed_s": round(now - begun, 1),
            })
            last_progress = now
    scores = np.asarray([[player.score for player in game.players] for game in games])
    for game in games:
        game.assert_invariants()
    for row in records:
        row["values"] = value_targets(scores[row["game"]], row["player"])
        row["value_valid"] = True
    own = np.asarray([
        games[i].players[agent_seats[i]].score for i in range(games_count)
    ])
    other = np.asarray([
        games[i].players[1 - agent_seats[i]].score for i in range(games_count)
    ])
    return records_to_replay(records), {
        "examples": len(records), "elapsed_s": time.time() - begun,
        "wins": int((own > other).sum()), "losses": int((own < other).sum()),
        "ties": int((own == other).sum()),
        "win_rate": float(((own > other).sum() + 0.5 * (own == other).sum()) / games_count),
        "margin": float((own - other).mean()),
    }


class RawAZ2Agent:
    def __init__(self, net: StructuredPolicyValueNet, device: torch.device):
        self.net = net; self.device = device

    @torch.inference_mode()
    def choose_many(self, games: list[AzulGame]) -> list[int]:
        if not games:
            return []
        obs = torch.from_numpy(np.stack([game.observation() for game in games])).to(self.device)
        masks = torch.from_numpy(np.stack([game.legal_mask() for game in games])).to(self.device)
        logits, _ = self.net(obs); logits.masked_fill_(~masks, -1e9)
        return logits.argmax(dim=1).cpu().tolist()

    def choose(self, game: AzulGame) -> int:
        return self.choose_many([game])[0]


class SearchAZ2Agent:
    """In-memory policy+search agent used for candidate promotion gates."""

    def __init__(
        self, net: StructuredPolicyValueNet, device: torch.device,
        simulations: int, heuristic_prior_weight: float,
        value_utility_weight: float = 1.0,
    ):
        self.search = FastBatchedPUCT(
            net, device, simulations=simulations, dirichlet_fraction=0.0,
            heuristic_prior_weight=heuristic_prior_weight,
            value_utility_weight=value_utility_weight,
        )
        self._pending_roots: dict[int, tuple[FastNode, int]] = {}

    def choose_many(self, games: list[AzulGame]) -> list[int]:
        if not games:
            return []
        reusable = []
        for game in games:
            pending = self._pending_roots.pop(id(game), None)
            reusable.append(
                self.search.recover_root_after_observed_moves(*pending, game)
                if pending is not None else None
            )
        policies, roots = self.search.search_many(
            games, add_noise=False, roots=reusable,
        )
        actions = [int(np.argmax(policy)) for policy in policies]
        for game, root, action in zip(games, roots, actions):
            self._pending_roots[id(game)] = (root, action)
        return actions

    def choose(self, game: AzulGame) -> int:
        return self.choose_many([game])[0]


def evaluate_agents(agent, opponent, games_count: int, seed: int) -> dict[str, float]:
    games = [AzulGame(seed + i) for i in range(games_count)]
    agent_seats = [i % 2 for i in range(games_count)]
    while any(not game.done for game in games):
        own_rows = [i for i, game in enumerate(games) if not game.done and game.current_player == agent_seats[i]]
        other_rows = [i for i, game in enumerate(games) if not game.done and game.current_player != agent_seats[i]]
        own_actions = agent.choose_many([games[i] for i in own_rows]) if own_rows else []
        for i, action in zip(own_rows, own_actions): games[i].step_fast(action)
        if hasattr(opponent, "choose_many"):
            other_actions = opponent.choose_many([games[i] for i in other_rows]) if other_rows else []
            for i, action in zip(other_rows, other_actions): games[i].step_fast(action)
        else:
            for i in other_rows: opponent_action = opponent.choose(games[i]); games[i].step_fast(opponent_action)
    own = np.asarray([game.players[agent_seats[i]].score for i, game in enumerate(games)])
    for game in games:
        game.assert_invariants()
    other = np.asarray([game.players[1 - agent_seats[i]].score for i, game in enumerate(games)])
    wins = int((own > other).sum()); losses = int((own < other).sum()); ties = games_count - wins - losses
    margins = own - other
    return {
        "games": games_count, "wins": wins, "losses": losses, "ties": ties,
        "win_rate": (wins + 0.5 * ties) / games_count,
        "score": float(own.mean()), "opponent_score": float(other.mean()),
        "margin": float(margins.mean()),
        "margin_stderr": float(margins.std(ddof=1) / np.sqrt(games_count)) if games_count > 1 else 0.0,
    }


def evaluate_score(agent, games_count: int, seed: int = 50_000) -> dict[str, float]:
    games = [AzulGame(seed + i) for i in range(games_count)]
    while any(not game.done for game in games):
        active = [game for game in games if not game.done]
        for game, action in zip(active, agent.choose_many(active)): game.step_fast(action)
    scores = np.asarray([[p.score for p in game.players] for game in games])
    for game in games:
        game.assert_invariants()
    return {"score_games": games_count, "mean_score": float(scores.mean()), "max_score": int(scores.max())}


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def train(
    output: str = "training_runs/az2", iterations: int = 5,
    games_per_iteration: int = 16, simulations: int = 128,
    teacher: str = "checkpoints/best_search_policy.pt", teacher_games: int = 64,
    bootstrap_epochs: int = 6, dagger_rounds: int = 0,
    dagger_games: int = 64, dagger_epochs: int = 8,
    epochs: int = 2, learning_rate: float = 3e-5,
    batch_size: int = 2048, replay_generations: int = 5,
    replay_max_examples: int = 60_000, gate_games: int = 64,
    score_games: int = 100, score_floor: float = 98.5,
    acceptance_rate: float = 0.52, seed: int = 20260712,
    resume: str | None = None, heuristic_prior_weight: float = 0.0,
    policy_weight: float = 1.0, search_gate_simulations: int = 0,
    value_utility_weight: float = 1.0,
    confirmation_games: int = 0,
    league_gate_games: int = 0,
    dirichlet_fraction: float = 0.25, value_weight: float = 1.0,
    value_margin_loss_weight: float = 1.0,
    value_win_loss_weight: float = 1.0,
    reset_optimizer: bool = False,
    policy_head_only: bool = False,
    value_mae_allowance: float = 1.01,
    value_reference_checkpoint: str | None = None,
) -> Path:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.backends.cuda.matmul.allow_tf32 = True
    start = 0
    out = Path(output); replay_dir = out / "replay"; out.mkdir(parents=True, exist_ok=True); replay_dir.mkdir(exist_ok=True)
    config = {
        "output": output, "iterations": iterations, "games_per_iteration": games_per_iteration,
        "simulations": simulations, "teacher": teacher, "teacher_games": teacher_games,
        "bootstrap_epochs": bootstrap_epochs, "dagger_rounds": dagger_rounds,
        "dagger_games": dagger_games, "dagger_epochs": dagger_epochs,
        "epochs": epochs, "learning_rate": learning_rate,
        "batch_size": batch_size, "replay_generations": replay_generations,
        "replay_max_examples": replay_max_examples, "gate_games": gate_games,
        "score_games": score_games, "score_floor": score_floor,
        "acceptance_rate": acceptance_rate, "seed": seed, "resume": resume,
        "heuristic_prior_weight": heuristic_prior_weight, "device": str(device),
        "policy_weight": policy_weight,
        "search_gate_simulations": search_gate_simulations,
        "value_utility_weight": value_utility_weight,
        "confirmation_games": confirmation_games,
        "league_gate_games": league_gate_games,
        "dirichlet_fraction": dirichlet_fraction,
        "value_weight": value_weight,
        "value_margin_loss_weight": value_margin_loss_weight,
        "value_win_loss_weight": value_win_loss_weight,
        "reset_optimizer": reset_optimizer,
        "policy_head_only": policy_head_only,
        "value_mae_allowance": value_mae_allowance,
        "value_reference_checkpoint": value_reference_checkpoint,
    }
    config_path = out / "config.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        mutable_on_resume = {"iterations", "resume"}
        mismatches = {
            key: (existing_config.get(key), value)
            for key, value in config.items()
            if key not in mutable_on_resume and existing_config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"run configuration differs from immutable config.json: {mismatches}")
    else:
        _write_json(config_path, config)
    _write_json(out / "pid.json", {
        "pid": os.getpid(), "started": time.time(), "command": sys.argv,
    })
    _write_json(out / "heartbeat.json", {
        "phase": "starting", "iteration": start if resume else 0,
        "timestamp": time.time(),
    })
    human = load_human_examples()
    validation = collect_teacher_examples(
        teacher, 64, device, seed - 2_000_000,
        temperature=1.0, hard_weight=0.0,
    )
    if resume:
        data = torch.load(resume, map_location=device, weights_only=False)
        net = load_az2_checkpoint(resume, device)
        if policy_head_only:
            for parameter in net.parameters():
                parameter.requires_grad_(False)
            for parameter in net.policy.parameters():
                parameter.requires_grad_(True)
        trainable = [parameter for parameter in net.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
        architecture_upgraded = any(
            key not in data["model"]
            for key in ("value_linear.weight", "value_mlp.0.weight")
        )
        if (
            data.get("optimizer") and not reset_optimizer
            and not architecture_upgraded and not policy_head_only
        ):
            optimizer.load_state_dict(data["optimizer"])
        for group in optimizer.param_groups: group["lr"] = learning_rate
        start = int(data.get("iteration", -1)) + 1
        save_az2_checkpoint(out / "latest.pt", net, optimizer, start - 1, config)
        save_az2_checkpoint(out / "best.pt", net, optimizer, start - 1, config)
    else:
        net = StructuredPolicyValueNet().to(device)
        legacy_data = torch.load(teacher, map_location=device, weights_only=False)
        net.baseline.load_state_dict(legacy_data["model"])
        net.baseline.eval()
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        # At temperature 1 with no hard mixing, the target distribution is
        # exactly the frozen baseline. Initial policy gradients are therefore
        # zero except for the deliberately low-weight human examples.
        teacher_data = collect_teacher_examples(
            teacher, teacher_games, device, seed - 1_000_000,
            temperature=1.0, hard_weight=0.0,
        )
        bootstrap_data = ReplayData.concat([teacher_data, human])
        bootstrap_metrics = train_epoch(
            net, optimizer, bootstrap_data, device, bootstrap_epochs, batch_size,
            policy_weight=0.0,
        )
        bootstrap_metrics.update(human_metrics(net, human, device))
        bootstrap_metrics.update({"phase": "bootstrap", "examples": len(bootstrap_data)})
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(bootstrap_metrics) + "\n")
        distill_replay = teacher_data
        for dagger_round in range(dagger_rounds):
            mix = 0.5 * (0.5 ** dagger_round)
            visited = collect_dagger_examples(
                net, teacher, dagger_games, device,
                seed - 500_000 + dagger_round * 100_000, teacher_mix=mix,
            )
            distill_replay = ReplayData.concat([distill_replay, visited])
            dagger_data = ReplayData.concat([distill_replay, human])
            dagger_metrics = train_epoch(
                net, optimizer, dagger_data, device, dagger_epochs, batch_size,
            )
            dagger_metrics.update(human_metrics(net, human, device))
            dagger_metrics.update({
                "phase": "dagger", "round": dagger_round + 1,
                "teacher_mix": mix, "examples": len(dagger_data),
            })
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dagger_metrics) + "\n")
        for group in optimizer.param_groups: group["lr"] = learning_rate
        save_az2_checkpoint(out / "bootstrap.pt", net, optimizer, -1, config)
        save_az2_checkpoint(out / "latest.pt", net, optimizer, -1, config)
        save_az2_checkpoint(out / "best.pt", net, optimizer, -1, config)
    legacy = NeuralAgent(teacher, device=str(device))
    value_reference = (
        value_metrics(
            load_az2_checkpoint(value_reference_checkpoint, device),
            validation, device,
        )
        if value_reference_checkpoint else None
    )
    league_path = out / "league.json"
    if league_path.exists():
        league = json.loads(league_path.read_text(encoding="utf-8"))
    else:
        league = {
            "entries": [
                *([{"id": "resume_incumbent", "path": resume, "kind": "az2"}] if resume else []),
                {"id": "legacy_search_policy", "path": teacher, "kind": "legacy"},
                {"id": "score_champion", "path": "checkpoints/best_score.pt", "kind": "legacy"},
            ],
            "accepted": [],
        }
    begun = time.time()
    for iteration in range(start, iterations):
        def report_progress(status: dict[str, int | float | str]) -> None:
            _write_json(out / "heartbeat.json", {
                "iteration": iteration, "timestamp": time.time(), **status,
            })

        generation, rollout = collect_search_self_play(
            net, games_per_iteration, device, seed + iteration * 100_000,
            simulations, heuristic_prior_weight=heuristic_prior_weight,
            value_utility_weight=value_utility_weight,
            dirichlet_fraction=dirichlet_fraction,
            progress=report_progress,
        )
        save_replay(replay_dir / f"generation_{iteration:04d}.npz", generation)
        replay = load_replay_window(replay_dir, replay_generations, replay_max_examples, seed + iteration)
        training_data = ReplayData.concat([replay, human])
        incumbent_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
        incumbent_optimizer = copy.deepcopy(optimizer.state_dict())
        incumbent = StructuredPolicyValueNet(net.hidden, net.layers, net.heads).to(device)
        incumbent.load_state_dict(incumbent_state); incumbent.eval()
        before_human = human_metrics(incumbent, human, device)
        before_value = value_metrics(incumbent, validation, device)
        report_progress({"phase": "training", "examples": len(training_data)})
        losses = train_epoch(
            net, optimizer, training_data, device, epochs, batch_size,
            policy_weight=policy_weight, value_weight=value_weight,
            value_head_weights=(
                1.0, 1.0, value_margin_loss_weight, value_win_loss_weight,
            ),
        )
        candidate_agent = RawAZ2Agent(net, device); incumbent_agent = RawAZ2Agent(incumbent, device)
        if search_gate_simulations:
            arena_candidate = SearchAZ2Agent(
                net, device, search_gate_simulations, heuristic_prior_weight,
                value_utility_weight,
            )
            arena_incumbent = SearchAZ2Agent(
                incumbent, device, search_gate_simulations, heuristic_prior_weight,
                value_utility_weight,
            )
            arena_mode = f"search_{search_gate_simulations}"
        else:
            arena_candidate = candidate_agent; arena_incumbent = incumbent_agent
            arena_mode = "raw"
        report_progress({"phase": "arena_gate", "games": gate_games})
        arena = evaluate_agents(arena_candidate, arena_incumbent, gate_games, seed + 700_000 + iteration * gate_games)
        confirmation = None
        if confirmation_games and arena["win_rate"] >= acceptance_rate:
            report_progress({"phase": "confirmation_arena", "games": confirmation_games})
            if search_gate_simulations:
                confirm_candidate = SearchAZ2Agent(
                    net, device, search_gate_simulations, heuristic_prior_weight,
                    value_utility_weight,
                )
                confirm_incumbent = SearchAZ2Agent(
                    incumbent, device, search_gate_simulations, heuristic_prior_weight,
                    value_utility_weight,
                )
            else:
                confirm_candidate = candidate_agent; confirm_incumbent = incumbent_agent
            confirmation = evaluate_agents(
                confirm_candidate, confirm_incumbent, confirmation_games,
                seed + 1_700_000 + iteration * confirmation_games,
            )
        league_gate = []
        preliminaries_pass = (
            arena["win_rate"] >= acceptance_rate
            and (confirmation is None or confirmation["win_rate"] >= acceptance_rate)
        )
        if league_gate_games and preliminaries_pass:
            for league_entry in league["accepted"][-3:]:
                opponent_path = Path(league_entry["path"])
                if not opponent_path.exists():
                    continue
                report_progress({
                    "phase": "opponent_league_gate", "games": league_gate_games,
                    "opponent_iteration": int(league_entry["iteration"]),
                })
                opponent_net = load_az2_checkpoint(opponent_path, device)
                if search_gate_simulations:
                    challenger = SearchAZ2Agent(
                        net, device, search_gate_simulations,
                        heuristic_prior_weight, value_utility_weight,
                    )
                    opponent_agent = SearchAZ2Agent(
                        opponent_net, device, search_gate_simulations,
                        heuristic_prior_weight, value_utility_weight,
                    )
                else:
                    challenger = candidate_agent
                    opponent_agent = RawAZ2Agent(opponent_net, device)
                result = evaluate_agents(
                    challenger, opponent_agent, league_gate_games,
                    seed + 2_700_000 + iteration * 10_000
                    + int(league_entry["iteration"]) * league_gate_games,
                )
                league_gate.append({
                    "opponent_iteration": int(league_entry["iteration"]),
                    "opponent_path": str(opponent_path), **result,
                })
        league_win_rate = (
            sum(row["wins"] + 0.5 * row["ties"] for row in league_gate)
            / sum(row["games"] for row in league_gate)
            if league_gate else None
        )
        heuristic_seed = seed + 800_000 + iteration * gate_games
        legacy_seed = seed + 900_000 + iteration * gate_games
        report_progress({"phase": "heuristic_baseline_gate", "games": gate_games})
        heuristic_gate_before = evaluate_agents(incumbent_agent, HeuristicAgent(123), gate_games, heuristic_seed)
        report_progress({"phase": "heuristic_candidate_gate", "games": gate_games})
        heuristic_gate = evaluate_agents(candidate_agent, HeuristicAgent(123), gate_games, heuristic_seed)
        report_progress({"phase": "legacy_baseline_gate", "games": gate_games})
        legacy_gate_before = evaluate_agents(incumbent_agent, legacy, gate_games, legacy_seed)
        report_progress({"phase": "legacy_candidate_gate", "games": gate_games})
        legacy_gate = evaluate_agents(candidate_agent, legacy, gate_games, legacy_seed)
        report_progress({"phase": "score_baseline_gate", "games": score_games})
        score_before = evaluate_score(incumbent_agent, score_games)
        report_progress({"phase": "score_candidate_gate", "games": score_games})
        score = evaluate_score(candidate_agent, score_games)
        after_human = human_metrics(net, human, device)
        after_value = value_metrics(net, validation, device)
        heuristic_allowance = max(
            1.0, 1.96 * np.hypot(
                heuristic_gate_before["margin_stderr"], heuristic_gate["margin_stderr"],
            ),
        )
        legacy_allowance = max(
            1.0, 1.96 * np.hypot(
                legacy_gate_before["margin_stderr"], legacy_gate["margin_stderr"],
            ),
        )
        accepted = (
            arena["win_rate"] >= acceptance_rate
            and (confirmation is None or confirmation["win_rate"] >= acceptance_rate)
            and (league_win_rate is None or league_win_rate >= 0.5)
            and score["mean_score"] >= max(score_floor, score_before["mean_score"] - 1.0)
            and heuristic_gate["margin"] >= heuristic_gate_before["margin"] - heuristic_allowance
            and legacy_gate["margin"] >= legacy_gate_before["margin"] - legacy_allowance
            and after_human["human_policy_nll"] <= before_human["human_policy_nll"] * 1.05
            and after_value["value_mae"] <= (
                (value_reference or before_value)["value_mae"]
                * value_mae_allowance
            )
        )
        candidate_path = out / f"candidate_{iteration:04d}.pt"
        save_az2_checkpoint(candidate_path, net, optimizer, iteration, config)
        if accepted:
            accepted_path = out / f"accepted_{iteration:04d}.pt"
            save_az2_checkpoint(accepted_path, net, optimizer, iteration, config)
            league["accepted"].append({
                "iteration": iteration, "path": str(accepted_path),
                "arena_win_rate": arena["win_rate"], "score": score["mean_score"],
            })
        else:
            net.load_state_dict(incumbent_state); optimizer.load_state_dict(incumbent_optimizer)
        row = {
            "iteration": iteration, **rollout, **losses,
            "arena_mode": arena_mode, "arena": arena,
            "confirmation_arena": confirmation,
            "opponent_league_gate": league_gate,
            "opponent_league_win_rate": league_win_rate,
            "heuristic_gate_before": heuristic_gate_before,
            "heuristic_gate": heuristic_gate, "heuristic_margin_allowance": heuristic_allowance,
            "legacy_gate_before": legacy_gate_before,
            "legacy_gate": legacy_gate, "legacy_margin_allowance": legacy_allowance,
            "score_gate_before": score_before, "score_gate": score,
            "human_before": before_human, "human_after": after_human,
            "value_validation_before": before_value,
            "value_validation_after": after_value,
            "value_validation_reference": value_reference,
            "replay_examples": len(replay), "training_examples": len(training_data),
            "accepted": accepted, "elapsed_s": round(time.time() - begun, 1),
        }
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        save_az2_checkpoint(out / "latest.pt", net, optimizer, iteration, config)
        if accepted: save_az2_checkpoint(out / "best.pt", net, optimizer, iteration, config)
        _write_json(out / "league.json", league)
        _write_json(out / "heartbeat.json", {
            "phase": "iteration_complete", "iteration": iteration,
            "timestamp": time.time(), "accepted": accepted,
            "latest": str(out / "latest.pt"),
        })
        try:
            print(json.dumps(row), flush=True)
        except BrokenPipeError:
            # A detached trainer must not lose a checkpoint merely because its
            # launching terminal closed. Files are the authoritative log.
            pass
    return out / "latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured stochastic PUCT self-play trainer")
    parser.add_argument("--output", default="training_runs/az2")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games-per-iteration", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--teacher", default="checkpoints/best_search_policy.pt")
    parser.add_argument("--teacher-games", type=int, default=64)
    parser.add_argument("--bootstrap-epochs", type=int, default=6)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-games", type=int, default=64)
    parser.add_argument("--dagger-epochs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--replay-generations", type=int, default=5)
    parser.add_argument("--replay-max-examples", type=int, default=60000)
    parser.add_argument("--gate-games", type=int, default=64)
    parser.add_argument("--score-games", type=int, default=100)
    parser.add_argument("--score-floor", type=float, default=98.5)
    parser.add_argument("--acceptance-rate", type=float, default=0.52)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--resume")
    parser.add_argument("--heuristic-prior-weight", type=float, default=0.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--search-gate-simulations", type=int, default=0)
    parser.add_argument("--value-utility-weight", type=float, default=1.0)
    parser.add_argument("--confirmation-games", type=int, default=0)
    parser.add_argument("--league-gate-games", type=int, default=0)
    parser.add_argument("--dirichlet-fraction", type=float, default=0.25)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--value-margin-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-win-loss-weight", type=float, default=1.0)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--policy-head-only", action="store_true")
    parser.add_argument("--value-mae-allowance", type=float, default=1.01)
    parser.add_argument("--value-reference-checkpoint")
    args = parser.parse_args()
    latest = train(
        args.output, args.iterations, args.games_per_iteration, args.simulations,
        args.teacher, args.teacher_games, args.bootstrap_epochs,
        args.dagger_rounds, args.dagger_games, args.dagger_epochs,
        args.epochs, args.learning_rate, args.batch_size, args.replay_generations,
        args.replay_max_examples, args.gate_games, args.score_games,
        args.score_floor, args.acceptance_rate, args.seed, args.resume,
        args.heuristic_prior_weight, args.policy_weight,
        args.search_gate_simulations, args.value_utility_weight,
        args.confirmation_games, args.league_gate_games,
        args.dirichlet_fraction, args.value_weight,
        args.value_margin_loss_weight, args.value_win_loss_weight,
        args.reset_optimizer,
        args.policy_head_only,
        args.value_mae_allowance, args.value_reference_checkpoint,
    )
    print(f"saved {latest}")


if __name__ == "__main__":
    main()
