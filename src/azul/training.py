from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .agents import HeuristicAgent, PolicyValueNet, RandomAgent, NeuralAgent, evaluate
from .game import AzulGame, NUM_ACTIONS, OBS_SIZE, STAR_BONUSES


def _masked_dist(net: nn.Module, obs: torch.Tensor, masks: torch.Tensor):
    logits, values = net(obs)
    logits = logits.masked_fill(~masks, -1e9)
    return torch.distributions.Categorical(logits=logits), values


def score_potential(player) -> float:
    """Smooth progress potential for redistributing sparse end-game bonuses.

    This potential is subtracted again at the terminal transition, so it changes
    credit timing rather than the undiscounted score-maximization objective.
    """
    potential = 0.0
    for color, star in enumerate(player.outer):
        potential += STAR_BONUSES[color] * (sum(star) / 6.0) ** 3
    potential += 12.0 * (sum(c >= 0 for c in player.center) / 6.0) ** 3
    for slot, bonus in enumerate((4, 8, 12, 16)):
        covered = sum(player.outer[s][slot] for s in range(6)) + (player.center[slot] >= 0)
        potential += bonus * (covered / 7.0) ** 3
    return potential


def outer_focus_potential(player) -> float:
    """Curriculum potential that rewards concentrating on one outer star."""
    return max(
        STAR_BONUSES[color] * (sum(star) / 6.0) ** 3
        for color, star in enumerate(player.outer)
    )


def multi_star_potential(player) -> float:
    """Curriculum potential for concentrated progress on up to three stars.

    The relaxed 60-net-tile board optimum completes the three highest-value
    outer stars. Taking only the best three progress terms avoids rewarding
    diffuse one-tile starts while opening a second/third-star behavior basin.
    """
    values = sorted((
        STAR_BONUSES[color] * (sum(star) / 6.0) ** 3
        for color, star in enumerate(player.outer)
    ), reverse=True)
    return sum(values[:3])


def collect_self_play(
    net: PolicyValueNet, games_count: int, device: torch.device, seed: int,
    objective: str = "competitive", gamma: float | None = None,
    gae_lambda: float | None = None, shaping_weight: float = 0.0,
    outer_curriculum_weight: float = 0.0, multi_star_curriculum_weight: float = 0.0,
):
    if objective not in {"competitive", "score", "team_score"}:
        raise ValueError(f"unknown objective: {objective}")
    # Absolute score is a six-round episodic objective. Full Monte Carlo credit
    # is intentional: λ=.97 would reduce a final bonus to ~6% over 90 decisions.
    is_absolute_score = objective in {"score", "team_score"}
    gamma = (1.0 if is_absolute_score else .995) if gamma is None else gamma
    gae_lambda = (1.0 if is_absolute_score else .95) if gae_lambda is None else gae_lambda
    games = [AzulGame(seed + i) for i in range(games_count)]
    trajectories = [[[], []] for _ in games]
    steps = 0
    net.eval()
    while any(not g.done for g in games):
        active = [(i, g) for i, g in enumerate(games) if not g.done]
        obs_np = np.stack([g.observation() for _, g in active])
        masks_np = np.stack([g.legal_mask() for _, g in active])
        obs = torch.from_numpy(obs_np).to(device)
        masks = torch.from_numpy(masks_np).to(device)
        with torch.inference_mode():
            dist, values = _masked_dist(net, obs, masks)
            actions = dist.sample()
            logps = dist.log_prob(actions)
        for row, (game_i, game) in enumerate(active):
            actor = game.current_player
            own_before = game.players[actor].score
            opp_before = game.players[1 - actor].score
            potential_before = score_potential(game.players[actor]) if shaping_weight else 0.0
            outer_before = outer_focus_potential(game.players[actor]) if outer_curriculum_weight else 0.0
            multi_before = multi_star_potential(game.players[actor]) if multi_star_curriculum_weight else 0.0
            record = {
                "obs": obs_np[row], "mask": masks_np[row], "action": int(actions[row]),
                "logp": float(logps[row]), "value": float(values[row]), "reward": 0.0,
            }
            game.step(int(actions[row]))
            own_after = game.players[actor].score
            opp_after = game.players[1 - actor].score
            if is_absolute_score:
                if objective == "team_score":
                    record["score_delta"] = own_after + opp_after - own_before - opp_before
                else:
                    record["score_delta"] = own_after - own_before
                record["reward"] = record["score_delta"] / 20.0
                if shaping_weight:
                    potential_after = score_potential(game.players[actor])
                    record["reward"] += shaping_weight * (potential_after - potential_before) / 20.0
                if outer_curriculum_weight:
                    outer_after = outer_focus_potential(game.players[actor])
                    # Intentionally not terminal-cancelled: this temporary
                    # curriculum breaks the no-outer-star local optimum. A later
                    # exact-score phase decides whether the behavior is retained.
                    record["reward"] += outer_curriculum_weight * (outer_after - outer_before) / 20.0
                if multi_star_curriculum_weight:
                    multi_after = multi_star_potential(game.players[actor])
                    # Exploration-only reward. Exact-score continuation and the
                    # fixed score gate decide whether the behavior is retained.
                    record["reward"] += multi_star_curriculum_weight * (multi_after - multi_before) / 20.0
            else:
                record["reward"] = (own_after - own_before - (opp_after - opp_before)) / 20.0
            trajectories[game_i][actor].append(record)
            steps += 1

    flat = []
    score_rows = []
    for game, pair in zip(games, trajectories):
        score_rows.append((game.players[0].score, game.players[1].score))
        for player, traj in enumerate(pair):
            if is_absolute_score:
                # Sum of episode rewards is exactly (final score - initial 5)/20.
                # Final scoring can occur on the opponent's last action, so add
                # any score delta not already observed on this player's actions.
                captured_delta = sum(t.get("score_delta", 0) for t in traj)
                if objective == "team_score":
                    residual_delta = sum(p.score for p in game.players) - 10 - captured_delta
                else:
                    residual_delta = game.players[player].score - 5 - captured_delta
                traj[-1]["reward"] += residual_delta / 20.0
                if shaping_weight:
                    traj[-1]["reward"] -= shaping_weight * score_potential(game.players[player]) / 20.0
            else:
                diff = game.players[player].score - game.players[1 - player].score
                outcome = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
                traj[-1]["reward"] += outcome + np.clip(diff / 50.0, -1, 1) + game.players[player].score / 250.0
            gae = 0.0
            next_value = 0.0
            for t in reversed(range(len(traj))):
                delta = traj[t]["reward"] + gamma * next_value - traj[t]["value"]
                gae = delta + gamma * gae_lambda * gae
                traj[t]["adv"] = gae
                traj[t]["return"] = gae + traj[t]["value"]
                next_value = traj[t]["value"]
            flat.extend(traj)
    player_count = games_count * 2
    outer_stars = sum(all(star) for game in games for p in game.players for star in p.outer)
    any_outer = sum(any(all(star) for star in p.outer) for game in games for p in game.players)
    center_stars = sum(all(c >= 0 for c in p.center) for game in games for p in game.players)
    score_array = np.asarray(score_rows)
    stats = {
        "outer_stars_per_player": outer_stars / player_count,
        "outer_completion_rate": any_outer / player_count,
        "center_stars_per_player": center_stars / player_count,
        "mean_combined_score": float(np.mean(score_array.sum(axis=1))),
        "max_combined_score": int(np.max(score_array.sum(axis=1))),
        "max_individual_score": int(np.max(score_array)),
    }
    return flat, steps, score_array, stats


def ppo_update(net, optimizer, records, device, epochs=3, batch_size=2048, entropy_coef=.015):
    net.train()
    obs = torch.from_numpy(np.stack([r["obs"] for r in records]))
    masks = torch.from_numpy(np.stack([r["mask"] for r in records]))
    actions = torch.tensor([r["action"] for r in records], dtype=torch.long)
    old_logp = torch.tensor([r["logp"] for r in records], dtype=torch.float32)
    returns = torch.tensor([r["return"] for r in records], dtype=torch.float32)
    adv = torch.tensor([r["adv"] for r in records], dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    losses = []
    n = len(records)
    for _ in range(epochs):
        for ids in torch.randperm(n).split(batch_size):
            ids = ids.to(device)
            dist, values = _masked_dist(net, obs[ids.cpu()].to(device), masks[ids.cpu()].to(device))
            logp = dist.log_prob(actions[ids.cpu()].to(device))
            ratio = (logp - old_logp[ids.cpu()].to(device)).exp()
            a = adv[ids.cpu()].to(device)
            policy_loss = -torch.minimum(ratio * a, ratio.clamp(.8, 1.2) * a).mean()
            value_loss = .5 * (values - returns[ids.cpu()].to(device)).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + .5 * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), .8)
            optimizer.step()
            losses.append((float(policy_loss), float(value_loss), float(entropy)))
    return np.mean(losses, axis=0)


def save_checkpoint(path: Path, net, optimizer, iteration: int, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": net.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration,
        "obs_size": OBS_SIZE, "hidden": 256, "config": config,
    }, path)


def train(
    iterations=20, games_per_iteration=64, seed=7, output="checkpoints", resume=None,
    batch_size=2048, ppo_epochs=3, objective="competitive", shaping_weight=0.0,
    outer_curriculum_weight=0.0, multi_star_curriculum_weight=0.0,
    learning_rate=None, entropy_coef=.015, reset_optimizer=False, reset_iteration=False,
    gamma=None, gae_lambda=None,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    net = PolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate or 3e-4, weight_decay=1e-4)
    start = 0
    if resume:
        data = torch.load(resume, map_location=device, weights_only=False)
        net.load_state_dict(data["model"])
        if not reset_optimizer:
            optimizer.load_state_dict(data["optimizer"])
        start = 0 if reset_iteration else data["iteration"] + 1
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    config = {
        "iterations": iterations, "games_per_iteration": games_per_iteration,
        "batch_size": batch_size, "ppo_epochs": ppo_epochs,
        "objective": objective, "shaping_weight": shaping_weight,
        "outer_curriculum_weight": outer_curriculum_weight,
        "multi_star_curriculum_weight": multi_star_curriculum_weight,
        "learning_rate": optimizer.param_groups[0]["lr"], "entropy_coef": entropy_coef,
        "gamma": gamma if gamma is not None else (1.0 if objective in {"score", "team_score"} else .995),
        "gae_lambda": gae_lambda if gae_lambda is not None else (1.0 if objective in {"score", "team_score"} else .95),
        "seed": seed, "device": str(device),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    history = []
    initial = out / "iteration_000.pt"
    if start == 0:
        save_checkpoint(initial, net, optimizer, 0, config)
    begun = time.time()
    for it in range(start, iterations):
        records, steps, scores, rollout_stats = collect_self_play(
            net, games_per_iteration, device, seed + it * 100_000,
            objective=objective, shaping_weight=shaping_weight,
            outer_curriculum_weight=outer_curriculum_weight,
            multi_star_curriculum_weight=multi_star_curriculum_weight,
            gamma=gamma, gae_lambda=gae_lambda,
        )
        p_loss, v_loss, entropy = ppo_update(
            net, optimizer, records, device, epochs=ppo_epochs, batch_size=batch_size,
            entropy_coef=entropy_coef,
        )
        checkpoint = out / f"iteration_{it + 1:03d}.pt"
        save_checkpoint(checkpoint, net, optimizer, it, config)
        row = {
            "iteration": it + 1, "steps": steps, "mean_score": float(scores.mean()),
            "median_score": float(np.median(scores)), "p10_score": float(np.percentile(scores, 10)),
            "p90_score": float(np.percentile(scores, 90)), "max_score": int(scores.max()),
            **rollout_stats,
            "policy_loss": float(p_loss), "value_loss": float(v_loss), "entropy": float(entropy),
            "elapsed_s": round(time.time() - begun, 1),
        }
        history.append(row)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    latest = out / "latest.pt"
    save_checkpoint(latest, net, optimizer, iterations - 1, config)
    return latest


def main():
    parser = argparse.ArgumentParser(description="GPU PPO self-play trainer")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games-per-iteration", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--resume")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--objective", choices=("competitive", "score", "team_score"), default="competitive")
    parser.add_argument("--shaping-weight", type=float, default=0.0)
    parser.add_argument("--outer-curriculum-weight", type=float, default=0.0)
    parser.add_argument("--multi-star-curriculum-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy-coef", type=float, default=.015)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--reset-iteration", action="store_true")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    args = parser.parse_args()
    latest = train(
        args.iterations, args.games_per_iteration, args.seed, args.output, args.resume,
        args.batch_size, args.ppo_epochs, args.objective, args.shaping_weight,
        args.outer_curriculum_weight,
        args.multi_star_curriculum_weight,
        args.learning_rate, args.entropy_coef, args.reset_optimizer, args.reset_iteration,
        args.gamma, args.gae_lambda,
    )
    print(f"saved {latest}")


if __name__ == "__main__":
    main()
