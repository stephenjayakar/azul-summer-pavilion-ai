"""Exact normal-board scoring bounds and relaxed two-player Pareto frontiers.

The board optimizer is exact: it chooses any subset of the 42 printed spaces,
orders placements optimally within each star, and accounts for every end-game
and architectural feature.  The game-level resource model is deliberately an
upper bound.  It charges a construction's printed placement costs and credits
the tiles returned by completed windows/statues/pillars, but relaxes color,
round, drafting-order, and supply-timing constraints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .game import AzulGame, BOARD_CLOCKWISE, COLORS, COSTS, STAR_BONUSES


FULL_MASK = (1 << 6) - 1
NUMBER_BONUSES = (4, 8, 12, 16)
N_STARS = 7
N_MASKS = 64
N_X = N_STARS * N_MASKS
FEATURES = tuple(AzulGame._features())
N_FEATURES = len(FEATURES)
FEATURE_OFFSET = N_X
NUMBER_OFFSET = FEATURE_OFFSET + N_FEATURES
N_VARIABLES = NUMBER_OFFSET + len(NUMBER_BONUSES)


def ring_score(mask: int) -> int:
    """Maximum placement score for a final subset of one six-space star."""
    if mask == FULL_MASK:
        return 21
    # Cut the cycle at an empty slot so a component crossing slot 6 -> slot 1
    # is counted as one run rather than two.
    empty = next(slot for slot in range(6) if not mask & (1 << slot))
    score = 0
    length = 0
    for offset in range(1, 7):
        slot = (empty + offset) % 6
        if mask & (1 << slot):
            length += 1
        elif length:
            score += length * (length + 1) // 2
            length = 0
    if length:
        score += length * (length + 1) // 2
    return score


def mask_cost(mask: int) -> int:
    return sum(cost for slot, cost in enumerate(COSTS) if mask & (1 << slot))


@dataclass(frozen=True)
class BoardSolution:
    score: int
    placement_score: int
    final_bonus: int
    gross_cost: int
    feature_tiles: int
    net_cost: int
    placed_spaces: int
    masks: tuple[int, ...]
    completed_outer_stars: tuple[str, ...]
    center_complete: bool
    completed_numbers: tuple[int, ...]
    completed_features: tuple[str, ...]


def _x(star: int, mask: int) -> int:
    return star * N_MASKS + mask


def _occupancy_coeff(row: np.ndarray, star: int, slot: int, value: float) -> None:
    bit = 1 << slot
    for mask in range(N_MASKS):
        if mask & bit:
            row[_x(star, mask)] += value


def _base_model() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    objective = np.zeros(N_VARIABLES)
    for star in range(N_STARS):
        for mask in range(N_MASKS):
            local = ring_score(mask)
            if mask == FULL_MASK:
                local += STAR_BONUSES[star] if star < 6 else 12
            objective[_x(star, mask)] = -local
    for slot, bonus in enumerate(NUMBER_BONUSES):
        objective[NUMBER_OFFSET + slot] = -bonus

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for star in range(N_STARS):
        row = np.zeros(N_VARIABLES)
        row[star * N_MASKS:(star + 1) * N_MASKS] = 1
        rows.append(row); lower.append(1); upper.append(1)

    # A feature variable is exactly the logical AND of all its printed cells.
    for feature_index, (_, _, cells, _) in enumerate(FEATURES):
        y = FEATURE_OFFSET + feature_index
        for star, slot in cells:
            row = np.zeros(N_VARIABLES); row[y] = 1
            _occupancy_coeff(row, star, slot, -1)
            rows.append(row); lower.append(-np.inf); upper.append(0)
        row = np.zeros(N_VARIABLES); row[y] = -1
        for star, slot in cells:
            _occupancy_coeff(row, star, slot, 1)
        rows.append(row); lower.append(-np.inf); upper.append(len(cells) - 1)

    # Number-set bonuses are ANDs across the six outer stars and center.
    for slot in range(len(NUMBER_BONUSES)):
        q = NUMBER_OFFSET + slot
        for star in range(N_STARS):
            row = np.zeros(N_VARIABLES); row[q] = 1
            _occupancy_coeff(row, star, slot, -1)
            rows.append(row); lower.append(-np.inf); upper.append(0)
        row = np.zeros(N_VARIABLES); row[q] = -1
        for star in range(N_STARS):
            _occupancy_coeff(row, star, slot, 1)
        rows.append(row); lower.append(-np.inf); upper.append(N_STARS - 1)

    return objective, np.stack(rows), np.asarray(lower), np.asarray(upper)


OBJECTIVE, BASE_A, BASE_LB, BASE_UB = _base_model()


def _resource_row(net: bool) -> np.ndarray:
    row = np.zeros(N_VARIABLES)
    for star in range(N_STARS):
        for mask in range(N_MASKS):
            row[_x(star, mask)] = mask_cost(mask)
    if net:
        for feature_index, (_, _, _, reward) in enumerate(FEATURES):
            row[FEATURE_OFFSET + feature_index] = -reward
    return row


GROSS_COST_ROW = _resource_row(False)
NET_COST_ROW = _resource_row(True)


def _decode_solution(values: np.ndarray) -> BoardSolution:
    masks = tuple(int(np.argmax(values[s * N_MASKS:(s + 1) * N_MASKS])) for s in range(N_STARS))
    placement = sum(ring_score(mask) for mask in masks)
    outer = tuple(COLORS[s] for s in range(6) if masks[s] == FULL_MASK)
    center_complete = masks[6] == FULL_MASK
    numbers = tuple(slot + 1 for slot in range(4) if all(mask & (1 << slot) for mask in masks))
    completed_features: list[str] = []
    feature_tiles = 0
    for kind, index, cells, reward in FEATURES:
        if all(masks[star] & (1 << slot) for star, slot in cells):
            completed_features.append(f"{kind}_{index}")
            feature_tiles += reward
    final_bonus = sum(STAR_BONUSES[s] for s in range(6) if masks[s] == FULL_MASK)
    final_bonus += 12 if center_complete else 0
    final_bonus += sum(NUMBER_BONUSES[slot - 1] for slot in numbers)
    gross = sum(mask_cost(mask) for mask in masks)
    return BoardSolution(
        score=5 + placement + final_bonus,
        placement_score=placement,
        final_bonus=final_bonus,
        gross_cost=gross,
        feature_tiles=feature_tiles,
        net_cost=gross - feature_tiles,
        placed_spaces=sum(mask.bit_count() for mask in masks),
        masks=masks,
        completed_outer_stars=outer,
        center_complete=center_complete,
        completed_numbers=numbers,
        completed_features=tuple(completed_features),
    )


def solve_board(*, net_budget: int | None = None, gross_budget: int | None = None) -> BoardSolution:
    """Return an exact maximum-scoring board under optional tile budgets."""
    rows = [BASE_A]
    lb = [BASE_LB]
    ub = [BASE_UB]
    if net_budget is not None:
        rows.append(NET_COST_ROW[None, :]); lb.append(np.asarray([-np.inf])); ub.append(np.asarray([net_budget]))
    if gross_budget is not None:
        rows.append(GROSS_COST_ROW[None, :]); lb.append(np.asarray([-np.inf])); ub.append(np.asarray([gross_budget]))
    result = milp(
        c=OBJECTIVE,
        integrality=np.ones(N_VARIABLES),
        bounds=Bounds(np.zeros(N_VARIABLES), np.ones(N_VARIABLES)),
        constraints=LinearConstraint(np.concatenate(rows), np.concatenate(lb), np.concatenate(ub)),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"board optimization failed: {result.message}")
    solution = _decode_solution(result.x)
    # The MILP objective excludes the five starting points.
    if solution.score != int(round(-result.fun)) + 5:
        raise AssertionError((solution.score, result.fun))
    return solution


def single_board_frontier(max_net_budget: int = 120) -> list[dict]:
    """Score-improving points on the exact net-tile/board-score frontier."""
    frontier: list[dict] = []
    best_score = -1
    for budget in range(max_net_budget + 1):
        solution = solve_board(net_budget=budget)
        if solution.score > best_score:
            row = asdict(solution)
            row["net_budget"] = budget
            frontier.append(row)
            best_score = solution.score
    return frontier


def _score_at(frontier: list[dict], budget: int) -> int:
    eligible = [row["score"] for row in frontier if row["net_budget"] <= budget]
    return max(eligible) if eligible else 5


def two_player_resource_frontier(single_frontier: list[dict], total_factory_tiles: int = 120) -> list[dict]:
    """Optimistic score-pair frontier under shared net construction resources.

    Factories expose 20 tiles in each of six two-player rounds. Architectural
    rewards are already credited in each board's net cost. Color, timing, and
    draft alternation are relaxed, so every returned point is an upper bound,
    not necessarily a legal-game construction.
    """
    candidates = []
    for first_budget in range(total_factory_tiles + 1):
        candidates.append({
            "p0_score": _score_at(single_frontier, first_budget),
            "p1_score": _score_at(single_frontier, total_factory_tiles - first_budget),
            "p0_net_budget": first_budget,
            "p1_net_budget": total_factory_tiles - first_budget,
        })
    frontier = []
    for row in candidates:
        dominated = any(
            other["p0_score"] >= row["p0_score"] and other["p1_score"] >= row["p1_score"]
            and (other["p0_score"] > row["p0_score"] or other["p1_score"] > row["p1_score"])
            for other in candidates
        )
        if not dominated:
            row["total_score"] = row["p0_score"] + row["p1_score"]
            frontier.append(row)
    unique = {(r["p0_score"], r["p1_score"]): r for r in frontier}
    return sorted(unique.values(), key=lambda row: row["p0_score"])


def theoretical_report() -> dict:
    ceiling = solve_board()
    frontier = single_board_frontier(120)
    pair_frontier = two_player_resource_frontier(frontier)
    return {
        "board_ceiling": asdict(ceiling),
        "interpretation": {
            "board_ceiling": "Exact maximum permitted by printed board scoring before game-flow feasibility.",
            "single_board_frontier": "Exact score maximum for each relaxed net tile budget.",
            "two_player_frontier": "Optimistic upper bound using 120 shared factory tiles plus feature rebates; color, rounds, supply timing, and alternating drafts are relaxed.",
        },
        "single_board_frontier": frontier,
        "two_player_resource_frontier": pair_frontier,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact board ceiling and relaxed score Pareto frontier")
    parser.add_argument("--output", type=Path, default=Path("theoretical_frontier.json"))
    args = parser.parse_args()
    report = theoretical_report()
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ceiling = report["board_ceiling"]
    pair = report["two_player_resource_frontier"]
    print(json.dumps({
        "board_ceiling": ceiling,
        "single_frontier_points": len(report["single_board_frontier"]),
        "two_player_frontier_points": len(pair),
        "max_relaxed_combined_score": max(row["total_score"] for row in pair),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
