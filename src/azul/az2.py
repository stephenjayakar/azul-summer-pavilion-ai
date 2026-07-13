"""Second-generation structured network and low-copy stochastic PUCT.

The search keeps one mutable scratch game per root. Every simulation restores a
compact immutable snapshot, applies a path with ``step_fast()``, batches leaf
evaluation, and restores on the next simulation. Future round draws are sampled
from a search-owned RNG rather than the live game's hidden future RNG stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .agents import HeuristicAgent, PolicyValueNet
from .game import AzulGame, GameSnapshot, NUM_ACTIONS, OBS_SIZE


VALUE_HEADS = 4


def value_targets(scores: Iterable[int], player: int) -> np.ndarray:
    scores = list(scores)
    own, other = scores[player], scores[1 - player]
    margin = float(np.clip((own - other) / 50.0, -1.0, 1.0))
    win = float(np.sign(own - other))
    return np.asarray((own / 200.0, other / 200.0, margin, win), dtype=np.float32)


def flip_values(values: np.ndarray) -> np.ndarray:
    """Convert [own, other, margin, win] to the other player's perspective."""
    return np.asarray((values[1], values[0], -values[2], -values[3]), dtype=np.float32)


class StructuredPolicyValueNet(nn.Module):
    """Entity-token transformer over factories, boards, pools, and players."""

    token_count = 38

    def __init__(
        self, hidden: int = 128, layers: int = 3, heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.global_proj = nn.Linear(40, hidden)
        self.factory_proj = nn.Linear(6, hidden)
        self.summary_proj = nn.Linear(16, hidden)
        self.outer_proj = nn.Linear(6, hidden)
        self.center_proj = nn.Linear(6, hidden)
        self.claim_proj = nn.Linear(6, hidden)
        self.type_embedding = nn.Embedding(6, hidden)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.token_count, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.baseline = PolicyValueNet()
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, NUM_ACTIONS),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, VALUE_HEADS),
        )
        # A direct calibrated path prevents scalar score/round signals from
        # being washed out by token pooling. Zero initialization keeps legacy
        # AZ2 checkpoints bit-for-bit equivalent until this path is trained.
        self.value_linear = nn.Linear(OBS_SIZE, VALUE_HEADS)
        self.value_mlp = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, VALUE_HEADS),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)
        nn.init.zeros_(self.value_linear.weight)
        nn.init.zeros_(self.value_linear.bias)
        nn.init.zeros_(self.value_mlp[-1].weight)
        nn.init.zeros_(self.value_mlp[-1].bias)

    @staticmethod
    def observation_slices() -> dict[str, tuple[int, int]]:
        return {
            "global_and_pools": (0, 38),
            "factories": (38, 68),
            "player_0": (68, 174),
            "player_1": (174, 280),
            "pending": (280, 282),
        }

    def tokenize(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[1] != OBS_SIZE:
            raise ValueError(f"expected [batch,{OBS_SIZE}] observations")
        batch = obs.shape[0]
        global_token = self.global_proj(torch.cat((obs[:, :38], obs[:, 280:282]), dim=1)).unsqueeze(1)
        factories = self.factory_proj(obs[:, 38:68].reshape(batch, 5, 6))
        summaries = []
        outers = []
        centers = []
        claims = []
        for base in (68, 174):
            summaries.append(self.summary_proj(obs[:, base:base + 16]).unsqueeze(1))
            outers.append(self.outer_proj(obs[:, base + 16:base + 52].reshape(batch, 6, 6)))
            centers.append(self.center_proj(obs[:, base + 52:base + 88].reshape(batch, 6, 6)))
            claims.append(self.claim_proj(obs[:, base + 88:base + 106].reshape(batch, 3, 6)))
        tokens = torch.cat((
            global_token, factories, torch.cat(summaries, dim=1),
            torch.cat(outers, dim=1), torch.cat(centers, dim=1),
            torch.cat(claims, dim=1),
        ), dim=1)
        type_ids = torch.tensor(
            [0] + [1] * 5 + [2] * 2 + [3] * 12 + [4] * 12 + [5] * 6,
            device=obs.device,
        )
        return tokens + self.type_embedding(type_ids).unsqueeze(0) + self.position_embedding

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.norm(self.encoder(self.tokenize(obs)))
        pooled = encoded[:, 0] + encoded[:, 1:].mean(dim=1)
        raw = self.value(pooled) + self.value_linear(obs) + self.value_mlp(obs)
        values = torch.cat((torch.sigmoid(raw[:, :2]), torch.tanh(raw[:, 2:])), dim=1)
        baseline_logits, _ = self.baseline(obs)
        return baseline_logits + self.policy(pooled), values


def save_az2_checkpoint(
    path: str | Path, net: StructuredPolicyValueNet,
    optimizer: torch.optim.Optimizer | None, iteration: int, config: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "az2", "model": net.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "iteration": iteration, "config": config,
        "hidden": net.hidden, "layers": net.layers, "heads": net.heads,
        "obs_size": OBS_SIZE, "num_actions": NUM_ACTIONS,
    }, path)
    return path


def load_az2_checkpoint(
    path: str | Path, device: str | torch.device = "cpu",
) -> StructuredPolicyValueNet:
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("format") != "az2":
        raise ValueError(f"not an az2 checkpoint: {path}")
    net = StructuredPolicyValueNet(
        hidden=data.get("hidden", 128), layers=data.get("layers", 3),
        heads=data.get("heads", 8),
    ).to(device)
    incompatible = net.load_state_dict(data["model"], strict=False)
    allowed_missing = {
        "value_linear.weight", "value_linear.bias",
        "value_mlp.0.weight", "value_mlp.0.bias",
        "value_mlp.1.weight", "value_mlp.1.bias",
        "value_mlp.3.weight", "value_mlp.3.bias",
    }
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise ValueError(
            f"incompatible az2 checkpoint: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    net.eval()
    return net


@dataclass(slots=True)
class FastNode:
    prior: float = 1.0
    to_play: int | None = None
    visits: int = 0
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros(VALUE_HEADS, dtype=np.float64))
    children: dict[int, "FastNode"] = field(default_factory=dict)
    expanded: bool = False
    state_key: bytes | None = None
    heuristic_blended: bool = False

    @property
    def value(self) -> np.ndarray:
        if not self.visits:
            return np.zeros(VALUE_HEADS, dtype=np.float32)
        return (self.value_sum / self.visits).astype(np.float32)


@dataclass(slots=True)
class RootContext:
    snapshot: GameSnapshot
    scratch: AzulGame
    node: FastNode


def public_state_key(game: AzulGame) -> bytes:
    return game.observation().tobytes() + game.legal_mask().tobytes()


class FastBatchedPUCT:
    """Low-copy PUCT with batched leaves and sampled future round draws."""

    def __init__(
        self, net: StructuredPolicyValueNet, device: torch.device,
        simulations: int = 128, c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3, dirichlet_fraction: float = 0.25,
        heuristic_prior_weight: float = 0.0, heuristic_temperature: float = 5.0,
        value_utility_weight: float = 1.0, seed: int = 0,
    ):
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if not 0 <= heuristic_prior_weight <= 1:
            raise ValueError("heuristic_prior_weight must be between 0 and 1")
        self.net = net
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self.heuristic_prior_weight = heuristic_prior_weight
        self.heuristic_temperature = heuristic_temperature
        self.value_utility_weight = value_utility_weight
        self.heuristic = HeuristicAgent(seed) if heuristic_prior_weight else None
        self.rng = np.random.default_rng(seed)
        self.chance_outcomes_seen: set[bytes] = set()

    @staticmethod
    def _perspective(values: np.ndarray, source_player: int, target_player: int) -> np.ndarray:
        return values if source_player == target_player else flip_values(values)

    def _utility(self, values: np.ndarray) -> float:
        score_diff = float(values[0] - values[1])
        utility = 0.45 * float(values[3]) + 0.35 * float(values[2]) + 0.20 * score_diff
        return self.value_utility_weight * utility

    def _select(self, parent: FastNode) -> tuple[int, FastNode]:
        if parent.to_play is None:
            raise RuntimeError("expanded parent has no player")
        root = np.sqrt(max(1, parent.visits))
        best = None
        selected = None
        for action, child in parent.children.items():
            q = 0.0
            if child.visits:
                values = self._perspective(child.value, child.to_play, parent.to_play)
                q = self._utility(values)
            u = self.c_puct * child.prior * root / (1 + child.visits)
            key = (q + u, child.prior, -action)
            if best is None or key > best:
                best = key
                selected = (action, child)
        if selected is None:
            raise RuntimeError("cannot select from leaf")
        return selected

    def _blend_heuristic(self, game: AzulGame, actions: list[int], priors: np.ndarray) -> np.ndarray:
        if not self.heuristic:
            return priors
        scores = np.asarray([self.heuristic._score(game, action) for action in actions], dtype=np.float64)
        scores = (scores - scores.max()) / self.heuristic_temperature
        teacher = np.exp(scores); teacher /= teacher.sum()
        keep = 1 - self.heuristic_prior_weight
        return keep * priors + self.heuristic_prior_weight * teacher

    def _expand(self, node: FastNode, game: AzulGame, logits: np.ndarray) -> None:
        actions = game.legal_actions()
        selected = logits[actions].astype(np.float64)
        selected -= selected.max()
        priors = np.exp(selected); priors /= priors.sum()
        node.children = {action: FastNode(prior=float(prior)) for action, prior in zip(actions, priors)}
        node.expanded = True
        node.to_play = game.current_player
        node.state_key = public_state_key(game)

    def _blend_root_heuristic(self, node: FastNode, game: AzulGame) -> None:
        """Apply the optional tactical teacher once, only when a node is a root.

        Scoring every newly expanded search leaf is both much more expensive and
        a different target from the successful root-policy improvement scheme.
        """
        if not self.heuristic or node.heuristic_blended or not node.children:
            return
        actions = list(node.children)
        priors = np.asarray([node.children[action].prior for action in actions], dtype=np.float64)
        priors = self._blend_heuristic(game, actions, priors)
        for action, prior in zip(actions, priors):
            node.children[action].prior = float(prior)
        node.heuristic_blended = True

    @staticmethod
    def _backup(path: list[FastNode], leaf_player: int, values: np.ndarray) -> None:
        for node in path:
            if node.to_play is None:
                node.to_play = leaf_player
            oriented = values if node.to_play == leaf_player else flip_values(values)
            node.visits += 1
            node.value_sum += oriented

    @torch.inference_mode()
    def _network(self, games: list[AzulGame]) -> tuple[np.ndarray, np.ndarray]:
        obs = torch.from_numpy(np.stack([game.observation() for game in games])).to(self.device)
        logits, values = self.net(obs)
        return logits.cpu().numpy(), values.cpu().numpy()

    def _new_contexts(
        self, games: list[AzulGame], roots: list[FastNode | None] | None = None,
    ) -> list[RootContext]:
        if roots is None:
            roots = [None] * len(games)
        if len(roots) != len(games):
            raise ValueError("roots must align with games")
        nodes: list[FastNode] = []
        new_rows = []
        for row, (game, candidate) in enumerate(zip(games, roots)):
            if candidate is not None and candidate.expanded and candidate.state_key == public_state_key(game):
                nodes.append(candidate)
            else:
                nodes.append(FastNode(to_play=game.current_player))
                new_rows.append(row)
        if new_rows:
            logits, values = self._network([games[row] for row in new_rows])
            for local, row in enumerate(new_rows):
                self._expand(nodes[row], games[row], logits[local])
                self._backup([nodes[row]], games[row].current_player, values[local])
        contexts = []
        for game, node in zip(games, nodes):
            self._blend_root_heuristic(node, game)
            if self.dirichlet_fraction:
                noise = self.rng.dirichlet([self.dirichlet_alpha] * len(node.children))
                keep = 1 - self.dirichlet_fraction
                for child, sample in zip(node.children.values(), noise):
                    child.prior = keep * child.prior + self.dirichlet_fraction * float(sample)
            contexts.append(RootContext(game.snapshot(), game.clone(), node))
        return contexts

    def search_many(
        self, games: list[AzulGame], add_noise: bool = True,
        roots: list[FastNode | None] | None = None,
    ) -> tuple[list[np.ndarray], list[FastNode]]:
        if not games:
            return [], []
        old_fraction = self.dirichlet_fraction
        if not add_noise:
            self.dirichlet_fraction = 0.0
        contexts = self._new_contexts(games, roots)
        self.dirichlet_fraction = old_fraction
        self.net.eval()

        for _ in range(self.simulations):
            paths: list[list[FastNode]] = []
            leaves: list[FastNode] = []
            leaf_games: list[AzulGame] = []
            chance_flags: list[bool] = []
            terminal_values: dict[int, tuple[int, np.ndarray]] = {}
            for row, context in enumerate(contexts):
                game = context.scratch
                game.restore(context.snapshot)
                # Determinize only future random draws; all currently visible
                # factories/pools remain exactly as observed at the root.
                game.rng.seed(int(self.rng.integers(0, 2**63 - 1)))
                node = context.node
                path = [node]
                crossed_chance = False
                while node.expanded and node.children:
                    action, child = self._select(node)
                    before_round = game.round
                    before_rng = game.rng.getstate()
                    game.step_fast(action)
                    if child.to_play is None:
                        child.to_play = game.current_player
                    node = child; path.append(node)
                    # Randomness enters both at new-round factory fills and at
                    # supply refills after the final pending bonus is taken.
                    crossed_chance = game.round != before_round or game.rng.getstate() != before_rng
                    if game.done or crossed_chance or not node.expanded:
                        break
                paths.append(path); leaves.append(node); leaf_games.append(game)
                chance_flags.append(crossed_chance)
                if game.done:
                    scores = [player.score for player in game.players]
                    terminal_values[row] = (game.current_player, value_targets(scores, game.current_player))
                elif crossed_chance:
                    self.chance_outcomes_seen.add(game.observation().tobytes())

            live_rows = [row for row in range(len(games)) if row not in terminal_values]
            if live_rows:
                logits, values = self._network([leaf_games[row] for row in live_rows])
                for local, row in enumerate(live_rows):
                    leaf = leaves[row]; game = leaf_games[row]
                    if not chance_flags[row] and not leaf.expanded:
                        self._expand(leaf, game, logits[local])
                    self._backup(paths[row], game.current_player, values[local])
            for row, (player, values) in terminal_values.items():
                self._backup(paths[row], player, values)

        policies = []
        roots = []
        for context in contexts:
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for action, child in context.node.children.items():
                policy[action] = child.visits
            if policy.sum() == 0:
                actions = list(context.node.children)
                policy[actions] = 1 / len(actions)
            else:
                policy /= policy.sum()
            policies.append(policy); roots.append(context.node)
        return policies, roots

    @staticmethod
    def advance_roots(
        roots: list[FastNode], actions: list[int], games_after: list[AzulGame],
    ) -> list[FastNode | None]:
        """Reuse selected deterministic subtrees when their public state matches."""
        advanced: list[FastNode | None] = []
        for root, action, game in zip(roots, actions, games_after):
            child = root.children.get(action)
            if child is not None and child.expanded and child.state_key == public_state_key(game):
                advanced.append(child)
            else:
                advanced.append(None)
        return advanced

    @staticmethod
    def recover_root_after_observed_moves(
        previous_root: FastNode, selected_action: int, game: AzulGame,
        max_depth: int = 4,
    ) -> FastNode | None:
        """Find a cached descendant matching a game advanced outside search.

        Evaluation and web agents do not control the opponent's move. Keeping
        the selected branch and matching the next public state recovers the
        same deterministic subtree reuse that self-play already receives.
        Chance-boundary children intentionally have no reusable state key.
        """
        selected = previous_root.children.get(selected_action)
        if selected is None:
            return None
        target = public_state_key(game)
        frontier = [selected]
        for _ in range(max_depth + 1):
            following = []
            for node in frontier:
                if node.expanded and node.state_key == target:
                    return node
                if node.expanded:
                    following.extend(node.children.values())
            if not following:
                break
            frontier = following
        return None


class AZ2Agent:
    def __init__(
        self, checkpoint: str | Path, device: str = "cpu", simulations: int = 128,
        heuristic_prior_weight: float = 0.0,
        value_utility_weight: float = 1.0,
    ):
        self.device = torch.device(device)
        self.net = load_az2_checkpoint(checkpoint, self.device)
        self.search = FastBatchedPUCT(
            self.net, self.device, simulations=simulations,
            dirichlet_fraction=0.0,
            heuristic_prior_weight=heuristic_prior_weight,
            value_utility_weight=value_utility_weight,
        )
        self._pending_roots: dict[int, tuple[FastNode, int]] = {}

    def choose_many(self, games: list[AzulGame]) -> list[int]:
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
