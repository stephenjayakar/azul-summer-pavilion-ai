"""AlphaZero-style search-guided self-play for competitive Azul.

Search and learning intentionally alternate instead of updating the network
inside a search.  Each self-play iteration freezes one policy/value snapshot,
uses batched PUCT to produce improved policy targets, finishes the games to get
score-margin value targets, and only then updates the network.

Unlike chess, an Azul action does not necessarily hand the turn to the other
player (placement, keeping, and bonuses can retain it).  Value propagation
therefore compares explicit player ids rather than negating at every edge.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .agents import HeuristicAgent, PolicyValueNet, load_network
from .game import AzulGame, NUM_ACTIONS
from .training import save_checkpoint


METHOD = "batched PUCT search-guided self-play"


@dataclass
class SearchNode:
    game: AzulGame | None
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def to_play(self) -> int:
        if self.game is None:
            raise RuntimeError("an unvisited lazy child has no materialized state")
        return self.game.current_player

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class BatchedPUCT:
    """Run independent PUCT trees while batching neural leaf evaluation."""

    def __init__(
        self, net: PolicyValueNet, device: torch.device, simulations: int = 32,
        c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25, value_scale: float = 50.0,
        seed: int = 0, heuristic_prior_weight: float = 0.0,
        heuristic_temperature: float = 5.0,
    ):
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if not 0.0 <= heuristic_prior_weight <= 1.0:
            raise ValueError("heuristic_prior_weight must be between 0 and 1")
        if heuristic_temperature <= 0:
            raise ValueError("heuristic_temperature must be positive")
        self.net = net
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self.value_scale = value_scale
        self.rng = np.random.default_rng(seed)
        self.heuristic_prior_weight = heuristic_prior_weight
        self.heuristic_temperature = heuristic_temperature
        self.heuristic = HeuristicAgent(seed) if heuristic_prior_weight else None

    def _terminal_value(self, node: SearchNode) -> float:
        if node.game is None:
            raise RuntimeError("cannot evaluate an unmaterialized node")
        scores = [player.score for player in node.game.players]
        margin = scores[node.to_play] - scores[1 - node.to_play]
        return float(np.clip(margin / self.value_scale, -1.0, 1.0))

    @torch.inference_mode()
    def _expand_many(self, nodes: list[SearchNode]) -> list[float]:
        values = [0.0] * len(nodes)
        if any(node.game is None for node in nodes):
            raise RuntimeError("all evaluated nodes must be materialized")
        live = [(i, node) for i, node in enumerate(nodes) if not node.game.done]
        for i, node in enumerate(nodes):
            if node.game.done:
                node.expanded = True
                values[i] = self._terminal_value(node)
        if not live:
            return values

        obs = torch.from_numpy(np.stack([node.game.observation() for _, node in live])).to(self.device)
        masks = torch.from_numpy(np.stack([node.game.legal_mask() for _, node in live])).to(self.device)
        logits, raw_values = self.net(obs)
        logits.masked_fill_(~masks, -1e9)
        priors = torch.softmax(logits, dim=1).cpu().numpy()
        predicted = torch.tanh(raw_values).cpu().numpy()
        for row, (original_i, node) in enumerate(live):
            for action in node.game.legal_actions():
                # Materialize only an edge PUCT actually selects.  Eagerly
                # cloning every legal child dominates simulator time.
                node.children[action] = SearchNode(None, prior=float(priors[row, action]))
            node.expanded = True
            values[original_i] = float(predicted[row])
        return values

    def _select_child(self, node: SearchNode) -> tuple[int, SearchNode]:
        sqrt_parent = np.sqrt(max(1, node.visit_count))
        best_key = None
        best_child = None
        for action, child in node.children.items():
            # Child values are stored for the player acting at the child.
            q = 0.0
            if child.visit_count:
                q = child.value if child.to_play == node.to_play else -child.value
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            key = (q + u, child.prior, -action)
            if best_key is None or key > best_key:
                best_key, best_child = key, child
        if best_child is None:
            raise RuntimeError("cannot select from an unexpanded or terminal node")
        return -best_key[2], best_child

    @staticmethod
    def _materialize_child(parent: SearchNode, action: int, child: SearchNode) -> None:
        if child.game is not None:
            return
        if parent.game is None:
            raise RuntimeError("cannot materialize from an absent parent state")
        child.game = parent.game.clone()
        child.game.step(action)

    @staticmethod
    def _backup(path: list[SearchNode], leaf_player: int, leaf_value: float) -> None:
        for node in path:
            node.visit_count += 1
            node.value_sum += leaf_value if node.to_play == leaf_player else -leaf_value

    def _add_root_noise(self, root: SearchNode) -> None:
        if not root.children or self.dirichlet_fraction <= 0:
            return
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(root.children))
        keep = 1.0 - self.dirichlet_fraction
        for child, sample in zip(root.children.values(), noise):
            child.prior = keep * child.prior + self.dirichlet_fraction * float(sample)

    def _blend_root_heuristic(self, root: SearchNode) -> None:
        if not self.heuristic or not root.children:
            return
        actions = list(root.children)
        scores = np.asarray([
            self.heuristic._score(root.game, action) for action in actions
        ], dtype=np.float64)
        scores = (scores - scores.max()) / self.heuristic_temperature
        heuristic_priors = np.exp(scores)
        heuristic_priors /= heuristic_priors.sum()
        keep = 1.0 - self.heuristic_prior_weight
        for action, prior in zip(actions, heuristic_priors):
            child = root.children[action]
            child.prior = keep * child.prior + self.heuristic_prior_weight * float(prior)

    def search_many(self, games: list[AzulGame], add_noise: bool = True) -> list[np.ndarray]:
        if not games:
            return []
        self.net.eval()
        roots = [SearchNode(game.clone()) for game in games]
        root_values = self._expand_many(roots)
        for root, value in zip(roots, root_values):
            self._backup([root], root.to_play, value)
            self._blend_root_heuristic(root)
            if add_noise:
                self._add_root_noise(root)

        for _ in range(self.simulations):
            paths: list[list[SearchNode]] = []
            leaves: list[SearchNode] = []
            for root in roots:
                node = root
                path = [node]
                while node.expanded and node.children:
                    action, child = self._select_child(node)
                    self._materialize_child(node, action, child)
                    node = child
                    path.append(node)
                paths.append(path)
                leaves.append(node)
            leaf_values = self._expand_many(leaves)
            for path, leaf, value in zip(paths, leaves, leaf_values):
                self._backup(path, leaf.to_play, value)

        policies = []
        for root in roots:
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for action, child in root.children.items():
                policy[action] = child.visit_count
            total = float(policy.sum())
            if not total:
                # This is only reachable with a future zero-simulation mode.
                legal = root.game.legal_actions()
                policy[legal] = 1.0 / len(legal)
            else:
                policy /= total
            policies.append(policy)
        return policies


class PUCTAgent:
    """Deployment agent combining a checkpoint policy/value net with PUCT."""

    def __init__(
        self, checkpoint: str | Path, device: str = "cpu", simulations: int = 32,
        c_puct: float = 1.5, value_scale: float = 50.0,
        heuristic_prior_weight: float = 0.0, heuristic_temperature: float = 5.0,
    ):
        self.device = torch.device(device)
        self.net = load_network(checkpoint, self.device)
        self.search = BatchedPUCT(
            self.net, self.device, simulations=simulations, c_puct=c_puct,
            dirichlet_fraction=0.0, value_scale=value_scale,
            heuristic_prior_weight=heuristic_prior_weight,
            heuristic_temperature=heuristic_temperature,
        )

    def choose(self, game: AzulGame) -> int:
        return self.choose_many([game])[0]

    def choose_many(self, games: list[AzulGame]) -> list[int]:
        policies = self.search.search_many(games, add_noise=False)
        return [int(np.argmax(policy)) for policy in policies]


def _sample_policy(policy: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 1e-6:
        return int(np.argmax(policy))
    scaled = np.power(policy, 1.0 / temperature, where=policy > 0, out=np.zeros_like(policy))
    scaled /= scaled.sum()
    return int(rng.choice(len(policy), p=scaled))


def collect_search_self_play(
    net: PolicyValueNet, games_count: int, device: torch.device, seed: int,
    simulations: int = 32, c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.25, temperature: float = 1.0,
    temperature_moves: int = 24, value_scale: float = 50.0,
    heuristic_prior_weight: float = 0.0, heuristic_temperature: float = 5.0,
) -> tuple[list[dict], np.ndarray, dict]:
    games = [AzulGame(seed + i) for i in range(games_count)]
    rows: list[dict] = []
    move_counts = [0] * games_count
    rng = np.random.default_rng(seed)
    search = BatchedPUCT(
        net, device, simulations, c_puct, dirichlet_alpha,
        dirichlet_fraction, value_scale, seed,
        heuristic_prior_weight, heuristic_temperature,
    )
    while any(not game.done for game in games):
        active = [(i, game) for i, game in enumerate(games) if not game.done]
        policies = search.search_many([game for _, game in active], add_noise=True)
        for (game_i, game), policy in zip(active, policies):
            actor = game.current_player
            temp = temperature if move_counts[game_i] < temperature_moves else 0.0
            rows.append({
                "obs": game.observation(), "mask": game.legal_mask(),
                "policy": policy, "player": actor, "game": game_i,
            })
            game.step(_sample_policy(policy, temp, rng))
            move_counts[game_i] += 1

    scores = np.asarray([(game.players[0].score, game.players[1].score) for game in games])
    for row in rows:
        own = scores[row["game"], row["player"]]
        other = scores[row["game"], 1 - row["player"]]
        row["value_target"] = float(np.clip((own - other) / value_scale, -1.0, 1.0))
    stats = {
        "mean_score": float(scores.mean()),
        "mean_combined_score": float(scores.sum(axis=1).mean()),
        "mean_margin_abs": float(np.abs(scores[:, 0] - scores[:, 1]).mean()),
        "max_score": int(scores.max()),
        "examples": len(rows),
    }
    return rows, scores, stats


@torch.inference_mode()
def collect_value_bootstrap_games(
    net: PolicyValueNet, games_count: int, device: torch.device, seed: int,
    value_scale: float = 50.0,
) -> tuple[list[dict], np.ndarray]:
    """Collect greedy policy games to calibrate an inherited value head."""
    games = [AzulGame(seed + i) for i in range(games_count)]
    rows: list[dict] = []
    net.eval()
    while any(not game.done for game in games):
        active = [(i, game) for i, game in enumerate(games) if not game.done]
        obs_np = np.stack([game.observation() for _, game in active])
        masks_np = np.stack([game.legal_mask() for _, game in active])
        obs = torch.from_numpy(obs_np).to(device)
        masks = torch.from_numpy(masks_np).to(device)
        logits, _ = net(obs)
        logits.masked_fill_(~masks, -1e9)
        actions = logits.argmax(dim=1).cpu().tolist()
        for row_i, ((game_i, game), action) in enumerate(zip(active, actions)):
            rows.append({
                "obs": obs_np[row_i], "player": game.current_player, "game": game_i,
            })
            game.step(action)
    scores = np.asarray([(game.players[0].score, game.players[1].score) for game in games])
    for row in rows:
        own = scores[row["game"], row["player"]]
        other = scores[row["game"], 1 - row["player"]]
        row["value_target"] = float(np.clip((own - other) / value_scale, -1.0, 1.0))
    return rows, scores


def value_bootstrap_update(
    net: PolicyValueNet, records: list[dict], device: torch.device,
    epochs: int = 4, batch_size: int = 4096, learning_rate: float = 3e-4,
) -> float:
    """Fit only the value head, leaving a proven inherited policy unchanged."""
    if not records:
        raise ValueError("cannot bootstrap without records")
    obs = torch.from_numpy(np.stack([row["obs"] for row in records]))
    targets = torch.tensor([row["value_target"] for row in records], dtype=torch.float32)
    optimizer = torch.optim.AdamW(net.value.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses = []
    net.train()
    for _ in range(epochs):
        for ids in torch.randperm(len(records)).split(batch_size):
            _, predicted = net(obs[ids].to(device))
            loss = F.mse_loss(torch.tanh(predicted), targets[ids].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.value.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
    return float(np.mean(losses))


@torch.inference_mode()
def arena_evaluate(
    candidate: PolicyValueNet, incumbent: PolicyValueNet, games_count: int,
    device: torch.device, seed: int = 80_000,
) -> dict[str, float]:
    """Alternating-seat, batched raw-policy gate between two networks."""
    games = [AzulGame(seed + i) for i in range(games_count)]
    candidate_seats = [i % 2 for i in range(games_count)]
    candidate.eval(); incumbent.eval()

    def choose_many(net: PolicyValueNet, rows: list[int]) -> list[int]:
        if not rows:
            return []
        obs = torch.from_numpy(np.stack([games[i].observation() for i in rows])).to(device)
        masks = torch.from_numpy(np.stack([games[i].legal_mask() for i in rows])).to(device)
        logits, _ = net(obs)
        logits.masked_fill_(~masks, -1e9)
        return logits.argmax(dim=1).cpu().tolist()

    while any(not game.done for game in games):
        candidate_rows = [
            i for i, game in enumerate(games)
            if not game.done and game.current_player == candidate_seats[i]
        ]
        incumbent_rows = [
            i for i, game in enumerate(games)
            if not game.done and game.current_player != candidate_seats[i]
        ]
        for i, action in zip(candidate_rows, choose_many(candidate, candidate_rows)):
            games[i].step(action)
        for i, action in zip(incumbent_rows, choose_many(incumbent, incumbent_rows)):
            games[i].step(action)

    own = np.asarray([
        game.players[candidate_seats[i]].score for i, game in enumerate(games)
    ])
    other = np.asarray([
        game.players[1 - candidate_seats[i]].score for i, game in enumerate(games)
    ])
    wins = int(np.sum(own > other)); losses = int(np.sum(own < other))
    ties = games_count - wins - losses
    return {
        "arena_games": games_count, "arena_wins": wins, "arena_losses": losses,
        "arena_ties": ties, "arena_win_rate": (wins + 0.5 * ties) / games_count,
        "arena_margin": float(np.mean(own - other)),
    }


def policy_value_update(
    net: PolicyValueNet, optimizer: torch.optim.Optimizer, records: list[dict],
    device: torch.device, epochs: int = 4, batch_size: int = 4096,
    value_weight: float = 1.0, entropy_coef: float = 0.002,
) -> tuple[float, float, float]:
    if not records:
        raise ValueError("cannot train without self-play records")
    net.train()
    obs = torch.from_numpy(np.stack([row["obs"] for row in records]))
    masks = torch.from_numpy(np.stack([row["mask"] for row in records]))
    targets = torch.from_numpy(np.stack([row["policy"] for row in records]))
    values = torch.tensor([row["value_target"] for row in records], dtype=torch.float32)
    losses = []
    for _ in range(epochs):
        for ids in torch.randperm(len(records)).split(batch_size):
            batch_obs = obs[ids].to(device)
            batch_masks = masks[ids].to(device)
            batch_targets = targets[ids].to(device)
            batch_values = values[ids].to(device)
            logits, predicted = net(batch_obs)
            logits.masked_fill_(~batch_masks, -1e9)
            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()
            policy_loss = -(batch_targets * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(torch.tanh(predicted), batch_values)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            loss = policy_loss + value_weight * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append((float(policy_loss), float(value_loss), float(entropy)))
    return tuple(float(x) for x in np.mean(losses, axis=0))


def train(
    iterations: int = 10, games_per_iteration: int = 64, simulations: int = 32,
    seed: int = 73, output: str = "checkpoints/alphazero", resume: str | None = None,
    epochs: int = 4, batch_size: int = 4096, learning_rate: float = 1e-4,
    c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.25, temperature: float = 1.0,
    temperature_moves: int = 24, value_scale: float = 50.0,
    replay_iterations: int = 3, value_weight: float = 1.0,
    entropy_coef: float = 0.002, bootstrap_games: int = 64,
    bootstrap_epochs: int = 4, arena_games: int = 64,
    acceptance_rate: float = 0.52, heuristic_prior_weight: float = 0.0,
    heuristic_temperature: float = 5.0,
) -> Path:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    net = PolicyValueNet().to(device)
    inherited_method = None
    resume_data = None
    if resume:
        resume_data = torch.load(resume, map_location=device, weights_only=False)
        net.load_state_dict(resume_data["model"])
        inherited_method = resume_data.get("config", {}).get("method")
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    if inherited_method == METHOD and resume_data and "optimizer" in resume_data:
        optimizer.load_state_dict(resume_data["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    config = {
        "method": METHOD, "iterations": iterations,
        "games_per_iteration": games_per_iteration, "simulations": simulations,
        "seed": seed, "resume": resume, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "c_puct": c_puct,
        "dirichlet_alpha": dirichlet_alpha, "dirichlet_fraction": dirichlet_fraction,
        "temperature": temperature, "temperature_moves": temperature_moves,
        "value_scale": value_scale, "replay_iterations": replay_iterations,
        "value_weight": value_weight, "entropy_coef": entropy_coef,
        "bootstrap_games": bootstrap_games, "bootstrap_epochs": bootstrap_epochs,
        "arena_games": arena_games, "acceptance_rate": acceptance_rate,
        "heuristic_prior_weight": heuristic_prior_weight,
        "heuristic_temperature": heuristic_temperature,
        "device": str(device), "training_during_search": False,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    replay: list[list[dict]] = []
    begun = time.time()
    # PPO score checkpoints use a positive, unbounded return value.  PUCT needs
    # a centered competitive value, so calibrate before allowing it to guide a
    # tree.  Existing search-guided checkpoints already have compatible heads.
    if resume and bootstrap_games and inherited_method != config["method"]:
        final_value_layer = net.value[-1]
        torch.nn.init.zeros_(final_value_layer.weight)
        torch.nn.init.zeros_(final_value_layer.bias)
        bootstrap_rows, bootstrap_scores = collect_value_bootstrap_games(
            net, bootstrap_games, device, seed - 10_000, value_scale,
        )
        bootstrap_loss = value_bootstrap_update(
            net, bootstrap_rows, device, bootstrap_epochs, batch_size,
        )
        bootstrap_row = {
            "phase": "value_bootstrap", "games": bootstrap_games,
            "examples": len(bootstrap_rows), "value_loss": bootstrap_loss,
            "mean_score": float(bootstrap_scores.mean()),
            "mean_margin_abs": float(np.abs(bootstrap_scores[:, 0] - bootstrap_scores[:, 1]).mean()),
            "elapsed_s": round(time.time() - begun, 1),
        }
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(bootstrap_row) + "\n")
        print(json.dumps(bootstrap_row), flush=True)
    for iteration in range(iterations):
        records, _, rollout_stats = collect_search_self_play(
            net, games_per_iteration, device, seed + iteration * 100_000,
            simulations, c_puct, dirichlet_alpha, dirichlet_fraction,
            temperature, temperature_moves, value_scale,
            heuristic_prior_weight, heuristic_temperature,
        )
        replay.append(records)
        replay = replay[-replay_iterations:]
        training_rows = [row for generation in replay for row in generation]
        incumbent_model = {key: value.detach().clone() for key, value in net.state_dict().items()}
        incumbent_optimizer = copy.deepcopy(optimizer.state_dict())
        policy_loss, value_loss, entropy = policy_value_update(
            net, optimizer, training_rows, device, epochs, batch_size,
            value_weight, entropy_coef,
        )
        arena_stats = {}
        accepted = True
        if arena_games:
            incumbent = PolicyValueNet().to(device)
            incumbent.load_state_dict(incumbent_model)
            arena_stats = arena_evaluate(
                net, incumbent, arena_games, device,
                seed=seed + 900_000 + iteration * arena_games,
            )
            accepted = arena_stats["arena_win_rate"] >= acceptance_rate
            if not accepted:
                net.load_state_dict(incumbent_model)
                optimizer.load_state_dict(incumbent_optimizer)
        row = {
            "iteration": iteration + 1, **rollout_stats,
            "replay_examples": len(training_rows), "policy_loss": policy_loss,
            "value_loss": value_loss, "entropy": entropy,
            **arena_stats, "accepted": accepted,
            "elapsed_s": round(time.time() - begun, 1),
        }
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        save_checkpoint(out / f"iteration_{iteration + 1:03d}.pt", net, optimizer, iteration, config)
    latest = out / "latest.pt"
    save_checkpoint(latest, net, optimizer, iterations - 1, config)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero-style batched PUCT self-play trainer")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", default="checkpoints/alphazero")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-fraction", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-moves", type=int, default=24)
    parser.add_argument("--value-scale", type=float, default=50.0)
    parser.add_argument("--replay-iterations", type=int, default=3)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--bootstrap-games", type=int, default=64)
    parser.add_argument("--bootstrap-epochs", type=int, default=4)
    parser.add_argument("--arena-games", type=int, default=64)
    parser.add_argument("--acceptance-rate", type=float, default=0.52)
    parser.add_argument("--heuristic-prior-weight", type=float, default=0.0)
    parser.add_argument("--heuristic-temperature", type=float, default=5.0)
    args = parser.parse_args()
    latest = train(
        args.iterations, args.games_per_iteration, args.simulations, args.seed,
        args.output, args.resume, args.epochs, args.batch_size, args.learning_rate,
        args.c_puct, args.dirichlet_alpha, args.dirichlet_fraction,
        args.temperature, args.temperature_moves, args.value_scale,
        args.replay_iterations, args.value_weight, args.entropy_coef,
        args.bootstrap_games, args.bootstrap_epochs,
        args.arena_games, args.acceptance_rate,
        args.heuristic_prior_weight, args.heuristic_temperature,
    )
    print(f"saved {latest}")


if __name__ == "__main__":
    main()
