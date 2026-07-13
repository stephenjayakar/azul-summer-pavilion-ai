from __future__ import annotations

from dataclasses import dataclass, field
import copy
import random
from typing import Iterable

import numpy as np

COLORS = ("purple", "green", "orange", "yellow", "blue", "red")
N_COLORS = 6
WILD_BY_ROUND = tuple(range(6))
STAR_BONUSES = (20, 18, 17, 16, 15, 14)  # same order as COLORS
COSTS = (1, 2, 3, 4, 5, 6)

# Fixed action space, so masks can be consumed by a neural policy. Nine factory
# IDs cover the official four-player maximum; ID 9 always means the center.
DRAFT_OFFSET = 0                  # 10 sources x 6 colors
CENTER_SOURCE = 9
OUTER_OFFSET = 60                 # star x slot x number of natural tiles used
CENTER_OFFSET = 276               # slot x placed color x natural tiles used
PASS_ACTION = 492
KEEP_OFFSET = 493                 # one per color
KEEP_FINISH = 499
BONUS_OFFSET = 500                # one per color
NUM_ACTIONS = 506
OBS_SIZE = 282

# Clockwise order on the printed normal-side player board. This is deliberately
# separate from COLORS/WILD_BY_ROUND: wild-round order is not board geometry.
BOARD_CLOCKWISE = (2, 5, 4, 3, 1, 0)  # orange, red, blue, yellow, green, purple


@dataclass(frozen=True)
class Action:
    kind: str
    source: int = -1
    color: int = -1
    star: int = -1
    slot: int = -1
    natural: int = -1


def encode_action(a: Action) -> int:
    if a.kind == "draft":
        return DRAFT_OFFSET + a.source * 6 + a.color
    if a.kind == "place_outer":
        return OUTER_OFFSET + (a.star * 36) + (a.slot * 6) + a.natural - 1
    if a.kind == "place_center":
        return CENTER_OFFSET + (a.slot * 36) + (a.color * 6) + a.natural - 1
    if a.kind == "pass":
        return PASS_ACTION
    if a.kind == "keep":
        return KEEP_OFFSET + a.color
    if a.kind == "keep_finish":
        return KEEP_FINISH
    if a.kind == "bonus":
        return BONUS_OFFSET + a.color
    raise ValueError(a)


def decode_action(action_id: int) -> Action:
    if 0 <= action_id < OUTER_OFFSET:
        source, color = divmod(action_id, 6)
        return Action("draft", source=source, color=color)
    if OUTER_OFFSET <= action_id < CENTER_OFFSET:
        x = action_id - OUTER_OFFSET
        star, x = divmod(x, 36)
        slot, n = divmod(x, 6)
        return Action("place_outer", star=star, slot=slot, natural=n + 1)
    if CENTER_OFFSET <= action_id < PASS_ACTION:
        x = action_id - CENTER_OFFSET
        slot, x = divmod(x, 36)
        color, n = divmod(x, 6)
        return Action("place_center", star=6, slot=slot, color=color, natural=n + 1)
    if action_id == PASS_ACTION:
        return Action("pass")
    if KEEP_OFFSET <= action_id < KEEP_FINISH:
        return Action("keep", color=action_id - KEEP_OFFSET)
    if action_id == KEEP_FINISH:
        return Action("keep_finish")
    if BONUS_OFFSET <= action_id < NUM_ACTIONS:
        return Action("bonus", color=action_id - BONUS_OFFSET)
    raise ValueError(f"invalid action id {action_id}")


@dataclass
class PlayerState:
    score: int = 5
    inventory: list[int] = field(default_factory=lambda: [0] * 6)
    stored: list[int] = field(default_factory=lambda: [0] * 6)
    # Outer stars are boolean; center contains a color index or -1.
    outer: list[list[bool]] = field(default_factory=lambda: [[False] * 6 for _ in range(6)])
    center: list[int] = field(default_factory=lambda: [-1] * 6)
    passed: bool = False
    keeping: bool = False
    keep_count: int = 0
    claimed: set[tuple[str, int]] = field(default_factory=set)

    def board_tile_count(self) -> int:
        return sum(map(sum, self.outer)) + sum(c >= 0 for c in self.center)


@dataclass(frozen=True, slots=True)
class PlayerSnapshot:
    score: int
    inventory: tuple[int, ...]
    stored: tuple[int, ...]
    outer: tuple[tuple[bool, ...], ...]
    center: tuple[int, ...]
    passed: bool
    keeping: bool
    keep_count: int
    claimed: frozenset[tuple[str, int]]


@dataclass(frozen=True, slots=True)
class GameSnapshot:
    round: int
    phase: str
    current_player: int
    start_player: int
    next_start_player: int | None
    players: tuple[PlayerSnapshot, ...]
    bag: tuple[int, ...]
    tower: tuple[int, ...]
    factories: tuple[tuple[int, ...], ...]
    center_pool: tuple[int, ...]
    supply: tuple[int, ...]
    pending_bonus: int
    pending_bonus_player: int
    final_score_breakdown: tuple[tuple[tuple[str, object], ...], ...]
    done: bool
    rng_state: object


class AzulGame:
    """Rules engine for the normal, colored side of 2–4 player Summer Pavilion.

    The six positions in every star are numbered 1..6 clockwise. Feature groups
    follow the board's repeating radial geometry. Random draws are reproducible.
    """

    def __init__(self, seed: int | None = None, num_players: int = 2):
        if num_players not in (2, 3, 4):
            raise ValueError("num_players must be 2, 3, or 4")
        self.num_players = num_players
        self.num_factories = 2 * num_players + 1
        self.seed = seed
        self.rng = random.Random(seed)
        self.round = 0
        self.phase = "draft"
        self.current_player = 0
        self.start_player = 0
        self.next_start_player: int | None = None
        self.players = [PlayerState() for _ in range(num_players)]
        self.bag = [22] * 6
        self.tower = [0] * 6
        self.factories = [[0] * 6 for _ in range(self.num_factories)]
        self.center_pool = [0] * 6
        self.supply = [0] * 6
        self.pending_bonus = 0
        self.pending_bonus_player = -1
        self.final_score_breakdown: list[dict] = []
        self.done = False
        self._fill_supply()
        self._fill_factories()

    @property
    def wild(self) -> int:
        return WILD_BY_ROUND[self.round]

    def clone(self) -> "AzulGame":
        return copy.deepcopy(self)

    def snapshot(self) -> GameSnapshot:
        """Capture mutable state for fast search apply/undo.

        This deliberately avoids copying the game object, Random instance, and
        dataclass scaffolding.  All captured containers are immutable, so one
        snapshot can be restored repeatedly during tree traversal.
        """
        players = tuple(PlayerSnapshot(
            score=p.score,
            inventory=tuple(p.inventory),
            stored=tuple(p.stored),
            outer=tuple(tuple(star) for star in p.outer),
            center=tuple(p.center),
            passed=p.passed,
            keeping=p.keeping,
            keep_count=p.keep_count,
            claimed=frozenset(p.claimed),
        ) for p in self.players)
        breakdown = tuple(
            tuple((key, copy.deepcopy(value)) for key, value in row.items())
            for row in self.final_score_breakdown
        )
        return GameSnapshot(
            round=self.round,
            phase=self.phase,
            current_player=self.current_player,
            start_player=self.start_player,
            next_start_player=self.next_start_player,
            players=players,
            bag=tuple(self.bag),
            tower=tuple(self.tower),
            factories=tuple(tuple(factory) for factory in self.factories),
            center_pool=tuple(self.center_pool),
            supply=tuple(self.supply),
            pending_bonus=self.pending_bonus,
            pending_bonus_player=self.pending_bonus_player,
            final_score_breakdown=breakdown,
            done=self.done,
            rng_state=self.rng.getstate(),
        )

    def restore(self, state: GameSnapshot) -> None:
        """Restore a state produced by :meth:`snapshot` in-place."""
        self.round = state.round
        self.phase = state.phase
        self.current_player = state.current_player
        self.start_player = state.start_player
        self.next_start_player = state.next_start_player
        self.bag[:] = state.bag
        self.tower[:] = state.tower
        if len(self.factories) != len(state.factories):
            self.factories = [list(factory) for factory in state.factories]
        else:
            for target, source in zip(self.factories, state.factories):
                target[:] = source
        self.center_pool[:] = state.center_pool
        self.supply[:] = state.supply
        self.pending_bonus = state.pending_bonus
        self.pending_bonus_player = state.pending_bonus_player
        self.done = state.done
        self.final_score_breakdown = [
            {key: copy.deepcopy(value) for key, value in row}
            for row in state.final_score_breakdown
        ]
        self.rng.setstate(state.rng_state)
        for player, saved in zip(self.players, state.players):
            player.score = saved.score
            player.inventory[:] = saved.inventory
            player.stored[:] = saved.stored
            for star, saved_star in zip(player.outer, saved.outer):
                star[:] = saved_star
            player.center[:] = saved.center
            player.passed = saved.passed
            player.keeping = saved.keeping
            player.keep_count = saved.keep_count
            player.claimed.clear()
            player.claimed.update(saved.claimed)

    def _draw_one(self) -> int | None:
        if sum(self.bag) == 0 and sum(self.tower):
            self.bag, self.tower = self.tower, [0] * 6
        total = sum(self.bag)
        if total == 0:
            return None
        pick = self.rng.randrange(total)
        for color, count in enumerate(self.bag):
            if pick < count:
                self.bag[color] -= 1
                return color
            pick -= count
        raise AssertionError("unreachable draw")

    def _fill_supply(self) -> None:
        while sum(self.supply) < 10:
            color = self._draw_one()
            if color is None:
                break
            self.supply[color] += 1

    def _fill_factories(self) -> None:
        self.factories = [[0] * 6 for _ in range(self.num_factories)]
        for factory in self.factories:
            for _ in range(4):
                color = self._draw_one()
                if color is None:
                    return
                factory[color] += 1

    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        p = self.players[self.current_player]
        if self.pending_bonus:
            return [BONUS_OFFSET + c for c, n in enumerate(self.supply) if n]
        if p.keeping:
            actions = [KEEP_FINISH]
            if p.keep_count < 4:
                actions += [KEEP_OFFSET + c for c, n in enumerate(p.inventory) if n]
            return actions
        if self.phase == "draft":
            actions: list[int] = []
            sources = list(enumerate(self.factories)) + [(CENTER_SOURCE, self.center_pool)]
            for source_id, source in sources:
                nonwild = [c for c, n in enumerate(source) if n and c != self.wild]
                if nonwild:
                    actions.extend(encode_action(Action("draft", source=source_id, color=c)) for c in nonwild)
                elif source[self.wild]:
                    actions.append(encode_action(Action("draft", source=source_id, color=self.wild)))
            return actions
        if self.phase == "place":
            actions = []
            for star in range(6):
                for slot, occupied in enumerate(p.outer[star]):
                    if not occupied:
                        actions.extend(self._payment_actions(star, slot, star))
            used_center_colors = set(c for c in p.center if c >= 0)
            for slot, existing in enumerate(p.center):
                if existing < 0:
                    for color in range(6):
                        if color not in used_center_colors:
                            actions.extend(self._payment_actions(6, slot, color))
            actions.append(PASS_ACTION)
            return actions
        raise AssertionError(self.phase)

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        mask[self.legal_actions()] = True
        return mask

    def _payment_actions(self, star: int, slot: int, color: int) -> list[int]:
        p = self.players[self.current_player]
        cost = COSTS[slot]
        if color == self.wild:
            naturals = [cost] if p.inventory[color] >= cost else []
        else:
            lo = max(1, cost - p.inventory[self.wild])
            hi = min(cost, p.inventory[color])
            naturals = range(lo, hi + 1) if lo <= hi else []
        kind = "place_center" if star == 6 else "place_outer"
        return [encode_action(Action(kind, star=star, slot=slot, color=color, natural=n)) for n in naturals]

    def step(self, action_id: int) -> None:
        legal = self.legal_actions()
        if action_id not in legal:
            raise ValueError(f"illegal action {action_id}: {decode_action(action_id)}")
        self._apply_action(action_id)
        self.assert_invariants()

    def step_fast(self, action_id: int) -> None:
        """Apply a trusted legal action without repeated validation.

        Search must obtain actions from ``legal_actions()`` before calling this
        method. Public gameplay should continue to use :meth:`step`.
        """
        self._apply_action(action_id)

    def _apply_action(self, action_id: int) -> None:
        action = decode_action(action_id)
        if action.kind == "draft":
            self._draft(action)
        elif action.kind.startswith("place"):
            self._place(action)
        elif action.kind == "pass":
            self.players[self.current_player].keeping = True
        elif action.kind == "keep":
            p = self.players[self.current_player]
            p.inventory[action.color] -= 1
            p.stored[action.color] += 1
            p.keep_count += 1
        elif action.kind == "keep_finish":
            self._finish_passing()
        elif action.kind == "bonus":
            self._take_bonus(action.color)

    def _draft(self, action: Action) -> None:
        source = self.center_pool if action.source == CENTER_SOURCE else self.factories[action.source]
        p = self.players[self.current_player]
        if action.color == self.wild:  # legal only when the source contains wilds exclusively
            taken = 1
            source[self.wild] -= 1
            p.inventory[self.wild] += 1
        else:
            taken = source[action.color]
            p.inventory[action.color] += taken
            source[action.color] = 0
            if source[self.wild]:
                source[self.wild] -= 1
                p.inventory[self.wild] += 1
                taken += 1
        if action.source != CENTER_SOURCE:
            for c in range(6):
                self.center_pool[c] += source[c]
                source[c] = 0
        elif self.next_start_player is None:
            self.next_start_player = self.current_player
            p.score = max(1, p.score - taken)

        if not any(map(sum, self.factories)) and not sum(self.center_pool):
            self.phase = "place"
            self.current_player = self.next_start_player if self.next_start_player is not None else self.start_player
            for player in self.players:
                player.passed = player.keeping = False
                player.keep_count = 0
        else:
            self.current_player = (self.current_player + 1) % self.num_players

    def _place(self, action: Action) -> None:
        p = self.players[self.current_player]
        color = action.color if action.kind == "place_center" else action.star
        cost = COSTS[action.slot]
        wild_used = cost - action.natural
        p.inventory[color] -= action.natural
        self.tower[color] += action.natural - 1
        if wild_used:
            p.inventory[self.wild] -= wild_used
            self.tower[self.wild] += wild_used
        if action.kind == "place_outer":
            p.outer[action.star][action.slot] = True
            star = action.star
        else:
            p.center[action.slot] = color
            star = 6
        p.score += self._connected_score(p, star, action.slot)

        reward_tiles = 0
        for kind, index, cells, reward in self._features():
            key = (kind, index)
            if key not in p.claimed and all(self._occupied(p, s, slot) for s, slot in cells):
                p.claimed.add(key)
                reward_tiles += reward
        if reward_tiles and sum(self.supply):
            self.pending_bonus = min(reward_tiles, sum(self.supply))
            self.pending_bonus_player = self.current_player
        else:
            self._advance_placement_turn()

    @staticmethod
    def _connected_score(p: PlayerState, star: int, slot: int) -> int:
        occupied = p.outer[star] if star < 6 else [c >= 0 for c in p.center]
        seen = {slot}
        frontier = [slot]
        while frontier:
            cur = frontier.pop()
            for nxt in ((cur - 1) % 6, (cur + 1) % 6):
                if occupied[nxt] and nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return len(seen)

    @staticmethod
    def _occupied(p: PlayerState, star: int, slot: int) -> bool:
        return p.outer[star][slot] if star < 6 else p.center[slot] >= 0

    @staticmethod
    def _features() -> Iterable[tuple[str, int, tuple[tuple[int, int], ...], int]]:
        """The 18 printed architectural features, transcribed by cost number.

        Slots are zero-based costs (slot 0 is printed cost 1). A window sits
        outside one color star at its 5/6 spaces. For every clockwise gap A→B,
        the statue touches A's 1/2 and B's 3/4. Each radial pillar touches one
        outer star's 2/3 spaces and its corresponding adjacent center-star pair.
        """
        for board_index, star in enumerate(BOARD_CLOCKWISE):
            nxt = BOARD_CLOCKWISE[(board_index + 1) % 6]
            yield "window", board_index, ((star, 4), (star, 5)), 3
            yield "statue", board_index, ((star, 0), (star, 1), (nxt, 2), (nxt, 3)), 2
            yield "pillar", board_index, (
                (star, 1), (star, 2), (6, (board_index - 1) % 6), (6, board_index)
            ), 1

    def _take_bonus(self, color: int) -> None:
        p = self.players[self.pending_bonus_player]
        self.supply[color] -= 1
        p.inventory[color] += 1
        self.pending_bonus -= 1
        if self.pending_bonus == 0 or sum(self.supply) == 0:
            self.pending_bonus = 0
            self.pending_bonus_player = -1
            self._fill_supply()
            self._advance_placement_turn()

    def _advance_placement_turn(self) -> None:
        nxt = self._next_unpassed(self.current_player)
        if nxt is None:
            self._end_round()
        else:
            self.current_player = nxt

    def _next_unpassed(self, after: int) -> int | None:
        for offset in range(1, self.num_players + 1):
            candidate = (after + offset) % self.num_players
            if not self.players[candidate].passed:
                return candidate
        return None

    def _finish_passing(self) -> None:
        p = self.players[self.current_player]
        discarded = sum(p.inventory)
        for c in range(6):
            self.tower[c] += p.inventory[c]
            p.inventory[c] = 0
        p.score = max(1, p.score - discarded)
        p.passed = True
        p.keeping = False
        nxt = self._next_unpassed(self.current_player)
        if nxt is None:
            self._end_round()
        else:
            self.current_player = nxt

    def _end_round(self) -> None:
        if self.round == 5:
            self._final_score()
            return
        self.round += 1
        self.start_player = self.next_start_player if self.next_start_player is not None else self.start_player
        self.next_start_player = None
        for p in self.players:
            for c in range(6):
                p.inventory[c] += p.stored[c]
                p.stored[c] = 0
            p.passed = p.keeping = False
            p.keep_count = 0
        self._fill_factories()
        self.phase = "draft"
        self.current_player = self.start_player

    def _final_score(self) -> None:
        self.final_score_breakdown = []
        for player_index, p in enumerate(self.players):
            score_before = p.score
            completed_stars = []
            for color in range(6):
                if all(p.outer[color]):
                    p.score += STAR_BONUSES[color]
                    completed_stars.append({
                        "color": COLORS[color],
                        "points": STAR_BONUSES[color],
                    })
            center_bonus = 0
            if all(c >= 0 for c in p.center):
                p.score += 12
                center_bonus = 12
            number_bonuses = []
            for slot, bonus in enumerate((4, 8, 12, 16)):
                covered = sum(p.outer[s][slot] for s in range(6)) + (p.center[slot] >= 0)
                if covered == 7:
                    p.score += bonus
                    number_bonuses.append({
                        "number": slot + 1,
                        "points": bonus,
                    })
            leftovers = sum(p.stored) + sum(p.inventory)
            p.score = max(1, p.score - leftovers)
            self.final_score_breakdown.append({
                "player": player_index,
                "score_before_final": score_before,
                "completed_stars": completed_stars,
                "center_bonus": center_bonus,
                "number_bonuses": number_bonuses,
                "leftover_penalty": leftovers,
                "final_bonus_total": (
                    sum(item["points"] for item in completed_stars)
                    + center_bonus
                    + sum(item["points"] for item in number_bonuses)
                ),
                "final_score": p.score,
            })
            for c in range(6):
                self.tower[c] += p.stored[c] + p.inventory[c]
                p.stored[c] = p.inventory[c] = 0
        self.done = True
        self.phase = "done"

    def winner(self) -> int | None:
        if not self.done:
            return None
        scores = [p.score for p in self.players]
        best = max(scores)
        winners = [i for i, score in enumerate(scores) if score == best]
        return winners[0] if len(winners) == 1 else None

    def action_description(self, action_id: int) -> str:
        a = decode_action(action_id)
        if a.kind == "draft":
            src = "center" if a.source == CENTER_SOURCE else f"factory {a.source + 1}"
            return f"take {COLORS[a.color]} from {src}"
        if a.kind == "place_outer":
            return f"place {COLORS[a.star]} on cost {a.slot + 1} using {a.natural} natural"
        if a.kind == "place_center":
            return f"place {COLORS[a.color]} in center cost {a.slot + 1} using {a.natural} natural"
        if a.kind == "pass":
            return "pass and choose up to four tiles to keep"
        if a.kind == "keep":
            return f"keep one {COLORS[a.color]}"
        if a.kind == "keep_finish":
            return "finish keeping (discard remaining tiles)"
        return f"take {COLORS[a.color]} bonus tile"

    def observation(self, perspective: int | None = None) -> np.ndarray:
        """Normalized, perfect-information state vector from one player's view."""
        if self.num_players != 2:
            raise ValueError("neural observations are intentionally two-player only")
        me = self.current_player if perspective is None else perspective
        order = (me, 1 - me)
        values: list[float] = []
        values += [self.round / 5, self.current_player == me, self.start_player == me]
        # The public first-player token is either still in the center, held by
        # the observer, or held by the opponent. Two bits distinguish all three.
        values += [self.next_start_player is not None, self.next_start_player == me]
        values += [int(self.phase == x) for x in ("draft", "place", "done")]
        values += [int(c == self.wild) for c in range(6)]
        values += [x / 22 for x in self.bag + self.tower + self.center_pool + self.supply]
        for f in self.factories:
            values += [x / 4 for x in f]
        for i in order:
            p = self.players[i]
            values += [p.score / 200, float(p.passed), float(p.keeping), p.keep_count / 4]
            values += [x / 22 for x in p.inventory + p.stored]
            values += [float(x) for star in p.outer for x in star]
            for slot in p.center:
                values += [float(slot == c) for c in range(6)]
            values += [float((kind, idx) in p.claimed) for kind in ("window", "statue", "pillar") for idx in range(6)]
        values += [self.pending_bonus / 6, self.pending_bonus_player == me]
        observation = np.asarray(values, dtype=np.float32)
        assert observation.shape == (OBS_SIZE,)
        return observation

    def assert_invariants(self) -> None:
        assert 0 <= self.round < 6
        assert all(x >= 0 for x in self.bag + self.tower + self.center_pool + self.supply)
        assert all(x >= 0 for f in self.factories for x in f)
        assert all(p.score >= 1 for p in self.players)
        for p in self.players:
            assert all(x >= 0 for x in p.inventory + p.stored)
            assert sum(p.stored) <= 4
            colors = [c for c in p.center if c >= 0]
            assert len(colors) == len(set(colors))
        total = sum(self.bag) + sum(self.tower) + sum(self.center_pool) + sum(self.supply)
        total += sum(map(sum, self.factories))
        total += sum(sum(p.inventory) + sum(p.stored) + p.board_tile_count() for p in self.players)
        assert total == 132, f"tile conservation failed: {total}"
