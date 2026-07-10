from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .game import (
    AzulGame,
    BONUS_OFFSET,
    CENTER_SOURCE,
    COLORS,
    KEEP_FINISH,
    KEEP_OFFSET,
    NUM_ACTIONS,
    OBS_SIZE,
    PASS_ACTION,
    STAR_BONUSES,
    decode_action,
)


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.policy = nn.Linear(hidden, NUM_ACTIONS)
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.body(obs)
        return self.policy(x), self.value(x).squeeze(-1)


def load_network(checkpoint: str | Path, device: str | torch.device = "cpu") -> PolicyValueNet:
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    net = PolicyValueNet(obs_size=data.get("obs_size", OBS_SIZE), hidden=data.get("hidden", 256)).to(device)
    net.load_state_dict(data["model"])
    net.eval()
    return net


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose(self, game: AzulGame) -> int:
        return self.rng.choice(game.legal_actions())


class NeuralAgent:
    def __init__(self, checkpoint: str | Path, device: str = "cpu", stochastic: bool = False):
        self.device = torch.device(device)
        self.net = load_network(checkpoint, self.device)
        self.stochastic = stochastic

    @torch.inference_mode()
    def choose(self, game: AzulGame) -> int:
        obs = torch.from_numpy(game.observation()).to(self.device).unsqueeze(0)
        mask = torch.from_numpy(game.legal_mask()).to(self.device).unsqueeze(0)
        logits, _ = self.net(obs)
        logits = logits.masked_fill(~mask, -1e9)
        if self.stochastic:
            return int(torch.distributions.Categorical(logits=logits).sample().item())
        return int(logits.argmax(dim=-1).item())

    @torch.inference_mode()
    def choose_many(self, games: list[AzulGame]) -> list[int]:
        if not games:
            return []
        obs = torch.from_numpy(np.stack([g.observation() for g in games])).to(self.device)
        masks = torch.from_numpy(np.stack([g.legal_mask() for g in games])).to(self.device)
        logits, _ = self.net(obs)
        logits.masked_fill_(~masks, -1e9)
        if self.stochastic:
            actions = torch.distributions.Categorical(logits=logits).sample()
        else:
            actions = logits.argmax(dim=-1)
        return [int(x) for x in actions.cpu()]


class HeuristicAgent:
    """Fast hand-written yardstick; it has no hidden information or look-ahead."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose(self, game: AzulGame) -> int:
        legal = game.legal_actions()
        scored = [(self._score(game, action), self.rng.random(), action) for action in legal]
        return max(scored)[2]

    def _score(self, game: AzulGame, action_id: int) -> float:
        a = decode_action(action_id)
        p = game.players[game.current_player]
        if a.kind == "bonus":
            # Wild now is flexible; otherwise prefer colors with a reachable cheap gap.
            useful = sum(not p.outer[a.color][s] for s in range(3))
            return 10 + (4 if a.color == game.wild else 0) + useful
        if a.kind == "keep_finish":
            return -sum(p.inventory) * 3
        if a.kind == "keep":
            next_wild = (game.round + 1) % 6
            return 20 + (8 if a.color == next_wild else 0) + p.inventory[a.color]
        if a.kind == "pass":
            return -100 - sum(p.inventory)
        if a.kind.startswith("place"):
            before_score = p.score
            before_claimed = len(p.claimed)
            clone = game.clone()
            clone.step(action_id)
            q = clone.players[game.current_player]
            gain = q.score - before_score
            features = len(q.claimed) - before_claimed
            color = a.color if a.kind == "place_center" else a.star
            cost = a.slot + 1
            wild_spent = cost - a.natural
            complete = all(q.outer[color]) if a.kind == "place_outer" else all(c >= 0 for c in q.center)
            # Connected placements and feature tiles dominate; preserve wilds when equivalent.
            return 100 + gain * 12 + features * 18 + complete * 16 - wild_spent * 1.5 - cost * .15
        if a.kind == "draft":
            source = game.center_pool if a.source == CENTER_SOURCE else game.factories[a.source]
            if a.color == game.wild:
                count = 1
            else:
                count = source[a.color] + int(source[game.wild] > 0)
            center_penalty = count if a.source == CENTER_SOURCE and game.next_start_player is None else 0
            color = a.color
            missing_costs = [slot + 1 for slot, x in enumerate(p.outer[color]) if not x]
            reachable = max((6 - abs((p.inventory[color] + count) - c) for c in missing_costs), default=0)
            wild_extra = int(a.color != game.wild and source[game.wild] > 0)
            return count * 7 + reachable + wild_extra * 5 - center_penalty * 2
        return 0


class ScoreHeuristicAgent(HeuristicAgent):
    """High-score curriculum policy that concentrates resources on one outer star."""

    def _target_star(self, game: AzulGame) -> int:
        p = game.players[game.current_player]
        # Acquired tiles and existing progress make the target state-dependent,
        # so self-play agents naturally diverge instead of contesting one color.
        values = [
            sum(p.outer[color]) * 12 + p.inventory[color] * 3 + STAR_BONUSES[color] * .2
            for color in range(6)
        ]
        # A deterministic self-play convention prevents both seats from
        # contesting the same 22-tile color before their boards diverge.
        preferred = 0 if game.current_player == 0 else 1  # purple / green
        values[preferred] += 6
        return int(np.argmax(values))

    def _score(self, game: AzulGame, action_id: int) -> float:
        a = decode_action(action_id)
        p = game.players[game.current_player]
        target = self._target_star(game)
        if a.kind == "bonus":
            next_wild = (game.round + 1) % 6
            return 50 + (35 if a.color == target else 0) + (12 if a.color == next_wild else 0)
        if a.kind == "keep":
            next_wild = (game.round + 1) % 6
            return 40 + (30 if a.color == target else 0) + (20 if a.color == next_wild else 0)
        if a.kind == "keep_finish":
            return -sum(p.inventory) * 4
        if a.kind == "pass":
            return -1000 - sum(p.inventory)
        if a.kind == "draft":
            source = game.center_pool if a.source == CENTER_SOURCE else game.factories[a.source]
            if a.color == game.wild:
                taken = [0] * 6; taken[game.wild] = 1
            else:
                taken = [0] * 6
                taken[a.color] = source[a.color]
                taken[game.wild] = int(source[game.wild] > 0)
            total = sum(taken)
            value = total * 5 + taken[target] * 18 + taken[game.wild] * 8
            value += taken[(game.round + 1) % 6] * 5
            if a.source == CENTER_SOURCE and game.next_start_player is None:
                value -= total * 2.5
            # Prefer drafts that make a missing target cost immediately reachable.
            available = p.inventory[target] + taken[target]
            value += max((8 - abs(available - (slot + 1)) for slot, x in enumerate(p.outer[target]) if not x), default=0)
            return value
        if a.kind.startswith("place"):
            color = a.color if a.kind == "place_center" else a.star
            before_score = p.score
            before_claimed = len(p.claimed)
            before_target = sum(p.outer[target])
            clone = game.clone(); clone.step(action_id)
            q = clone.players[game.current_player]
            gain = q.score - before_score
            features = len(q.claimed) - before_claimed
            cost = a.slot + 1
            wild_spent = cost - a.natural
            score = 100 + gain * 12 + features * 24 - wild_spent
            if a.kind == "place_outer" and color == target:
                after_target = sum(q.outer[target])
                progress_gain = STAR_BONUSES[target] * (
                    (after_target / 6) ** 3 - (before_target / 6) ** 3
                )
                score += 45 + progress_gain * 18
                if after_target == 6:
                    score += 160
            elif a.kind == "place_center":
                score -= 12
            # Cheap-number coverage remains valuable around the star plan.
            if a.slot < 3:
                covered_before = sum(p.outer[s][a.slot] for s in range(6)) + (p.center[a.slot] >= 0)
                score += covered_before * 3
            # Do not burn a target-colored wild on unrelated placements.
            if game.wild == target and color != target:
                score -= wild_spent * 15
            return score
        return super()._score(game, action_id)


class MultiStarHeuristicAgent(HeuristicAgent):
    """Exploration teacher that deliberately develops two outer stars."""

    @staticmethod
    def _targets(game: AzulGame) -> tuple[int, int]:
        # Disjoint conventions reduce draft contention during teacher self-play.
        return (0, 2) if game.current_player == 0 else (1, 5)

    def _score(self, game: AzulGame, action_id: int) -> float:
        a = decode_action(action_id); p = game.players[game.current_player]
        targets = self._targets(game)
        if a.kind == "bonus":
            return 50 + (35 if a.color in targets else 0) + (12 if a.color == (game.round + 1) % 6 else 0)
        if a.kind == "keep":
            return 40 + (30 if a.color in targets else 0) + (20 if a.color == (game.round + 1) % 6 else 0)
        if a.kind == "keep_finish":
            return -sum(p.inventory) * 4
        if a.kind == "pass":
            return -1000 - sum(p.inventory)
        if a.kind == "draft":
            source = game.center_pool if a.source == CENTER_SOURCE else game.factories[a.source]
            taken = [0] * 6
            if a.color == game.wild:
                taken[game.wild] = 1
            else:
                taken[a.color] = source[a.color]
                taken[game.wild] = int(source[game.wild] > 0)
            total = sum(taken)
            value = total * 5 + sum(taken[t] for t in targets) * 18 + taken[game.wild] * 8
            value += taken[(game.round + 1) % 6] * 5
            if a.source == CENTER_SOURCE and game.next_start_player is None:
                value -= total * 2.5
            for target in targets:
                available = p.inventory[target] + taken[target]
                value += max((6 - abs(available - (slot + 1)) for slot, x in enumerate(p.outer[target]) if not x), default=0)
            return value
        if a.kind.startswith("place"):
            color = a.color if a.kind == "place_center" else a.star
            before_score = p.score; before_claimed = len(p.claimed)
            before_progress = sum(p.outer[color]) if color in targets else 0
            clone = game.clone(); clone.step(action_id); q = clone.players[game.current_player]
            gain = q.score - before_score; features = len(q.claimed) - before_claimed
            cost = a.slot + 1; wild_spent = cost - a.natural
            score = 100 + gain * 12 + features * 24 - wild_spent
            if a.kind == "place_outer" and color in targets:
                after = sum(q.outer[color])
                progress_gain = STAR_BONUSES[color] * ((after / 6) ** 3 - (before_progress / 6) ** 3)
                score += 50 + progress_gain * 20 + (200 if after == 6 else 0)
                # Once one star is complete, strongly favor the unfinished target.
                other = targets[0] if color == targets[1] else targets[1]
                if all(p.outer[other]) and after < 6:
                    score += 60
            elif a.kind == "place_center":
                score -= 15
            elif a.slot >= 4:
                score -= 35
            if a.slot < 4:
                covered = sum(p.outer[s][a.slot] for s in range(6)) + (p.center[a.slot] >= 0)
                score += covered * 3
            return score
        return super()._score(game, action_id)


class HybridAgent:
    """Strong play mode: tactical guardrails with neural tie-breaking.

    The heuristic prevents obvious long-horizon resource mistakes; among moves
    whose strategic scores are effectively tied, the learned policy chooses.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu", tolerance: float = .5):
        self.neural = NeuralAgent(checkpoint, device=device)
        self.heuristic = HeuristicAgent(0)
        self.tolerance = tolerance

    @torch.inference_mode()
    def choose(self, game: AzulGame) -> int:
        legal = game.legal_actions()
        strategic = np.asarray([self.heuristic._score(game, action) for action in legal])
        candidates = np.asarray(legal)[strategic >= strategic.max() - self.tolerance]
        if len(candidates) == 1:
            return int(candidates[0])
        obs = torch.from_numpy(game.observation()).to(self.neural.device).unsqueeze(0)
        logits, _ = self.neural.net(obs)
        candidate_ids = torch.as_tensor(candidates, dtype=torch.long, device=self.neural.device)
        return int(candidate_ids[logits[0, candidate_ids].argmax()].item())


class RolloutAgent:
    """Policy-improvement agent using deterministic full-game neural rollouts.

    Search is normally limited to the final round, where terminal bonuses make
    one-step policy logits least reliable and complete rollouts are cheapest.
    Candidate games are evaluated in one neural batch across all live boards.
    """

    def __init__(
        self, checkpoint: str | Path, device: str = "cpu", top_k: int = 6,
        search_from_round: int = 5, objective: str = "own",
    ):
        if objective not in {"own", "team"}:
            raise ValueError(f"unknown rollout objective: {objective}")
        self.neural = NeuralAgent(checkpoint, device=device)
        self.top_k = top_k
        self.search_from_round = search_from_round
        self.objective = objective

    def choose(self, game: AzulGame) -> int:
        return self.choose_many([game])[0]

    @torch.inference_mode()
    def choose_many(self, games: list[AzulGame]) -> list[int]:
        if not games:
            return []
        base_actions = self.neural.choose_many(games)
        search_rows = [i for i, game in enumerate(games) if game.round >= self.search_from_round]
        if not search_rows or self.top_k <= 1:
            return base_actions
        search_games = [games[i] for i in search_rows]
        obs = torch.from_numpy(np.stack([g.observation() for g in search_games])).to(self.neural.device)
        masks = torch.from_numpy(np.stack([g.legal_mask() for g in search_games])).to(self.neural.device)
        logits, _ = self.neural.net(obs); logits.masked_fill_(~masks, -1e9)

        candidates: list[AzulGame] = []
        metadata: list[tuple[int, int, int, float]] = []
        for local_i, game in enumerate(search_games):
            count = min(self.top_k, len(game.legal_actions()))
            ids = torch.topk(logits[local_i], count).indices.tolist()
            actor = game.current_player
            for action in ids:
                clone = game.clone(); clone.step(action)
                metadata.append((local_i, actor, action, float(logits[local_i, action])))
                candidates.append(clone)

        while any(not candidate.done for candidate in candidates):
            active_ids = [i for i, candidate in enumerate(candidates) if not candidate.done]
            actions = self.neural.choose_many([candidates[i] for i in active_ids])
            for candidate_i, action in zip(active_ids, actions):
                candidates[candidate_i].step(action)

        best: dict[int, tuple[int, float, int]] = {}
        for candidate, (local_i, actor, action, logit) in zip(candidates, metadata):
            utility = (
                sum(player.score for player in candidate.players)
                if self.objective == "team" else candidate.players[actor].score
            )
            key = (utility, logit, action)
            if local_i not in best or key > best[local_i]:
                best[local_i] = key
        for local_i, original_i in enumerate(search_rows):
            base_actions[original_i] = best[local_i][2]
        return base_actions


def play_game(agent0, agent1, seed: int) -> AzulGame:
    game = AzulGame(seed)
    agents = (agent0, agent1)
    while not game.done:
        game.step(agents[game.current_player].choose(game))
    return game


def evaluate(agent0, agent1, games: int = 100, seed: int = 10_000) -> dict[str, float]:
    wins = losses = ties = 0
    scores = [[], []]
    # Swap seats to remove first-player bias while keeping the report agent0-centric.
    for i in range(games):
        swapped = i % 2 == 1
        game = play_game(agent1, agent0, seed + i) if swapped else play_game(agent0, agent1, seed + i)
        s0, s1 = (game.players[1].score, game.players[0].score) if swapped else (game.players[0].score, game.players[1].score)
        scores[0].append(s0)
        scores[1].append(s1)
        wins += s0 > s1
        losses += s0 < s1
        ties += s0 == s1
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": (wins + .5 * ties) / games,
        "score": float(np.mean(scores[0])),
        "opponent_score": float(np.mean(scores[1])),
        "margin": float(np.mean(np.asarray(scores[0]) - np.asarray(scores[1]))),
    }


def evaluate_neural_batched(agent: NeuralAgent, opponent, games: int = 100, seed: int = 10_000) -> dict[str, float]:
    """Alternating-seat evaluation with all neural turns inferred as one batch."""
    states = [AzulGame(seed + i) for i in range(games)]
    neural_seats = [i % 2 for i in range(games)]
    while any(not g.done for g in states):
        neural_rows = [i for i, g in enumerate(states) if not g.done and g.current_player == neural_seats[i]]
        actions = agent.choose_many([states[i] for i in neural_rows])
        for i, action in zip(neural_rows, actions):
            states[i].step(action)
        for i, g in enumerate(states):
            if not g.done and g.current_player != neural_seats[i]:
                g.step(opponent.choose(g))
    wins = losses = ties = 0
    own, other = [], []
    for i, g in enumerate(states):
        s0 = g.players[neural_seats[i]].score
        s1 = g.players[1 - neural_seats[i]].score
        own.append(s0); other.append(s1)
        wins += s0 > s1; losses += s0 < s1; ties += s0 == s1
    return {
        "games": games, "wins": wins, "losses": losses, "ties": ties,
        "win_rate": (wins + .5 * ties) / games,
        "score": float(np.mean(own)), "opponent_score": float(np.mean(other)),
        "margin": float(np.mean(np.asarray(own) - np.asarray(other))),
    }


def evaluate_self_play_batched(agent: NeuralAgent, games: int = 100, seed: int = 50_000) -> dict[str, float]:
    """Fixed-seed score benchmark with the same neural policy in every seat."""
    states = [AzulGame(seed + i) for i in range(games)]
    while any(not game.done for game in states):
        active = [game for game in states if not game.done]
        actions = agent.choose_many(active)
        for game, action in zip(active, actions):
            game.step(action)
    scores = np.asarray([player.score for game in states for player in game.players], dtype=np.float32)
    outer_stars = sum(all(star) for game in states for player in game.players for star in player.outer)
    center_stars = sum(all(c >= 0 for c in player.center) for game in states for player in game.players)
    return {
        "games": games,
        "player_scores": int(scores.size),
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "p10_score": float(np.percentile(scores, 10)),
        "p90_score": float(np.percentile(scores, 90)),
        "max_score": int(scores.max()),
        "outer_stars_per_player": outer_stars / scores.size,
        "center_stars_per_player": center_stars / scores.size,
    }
