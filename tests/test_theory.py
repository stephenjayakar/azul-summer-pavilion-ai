import itertools

from azul.theory import ring_score, solve_board, single_board_frontier, two_player_resource_frontier


def test_ring_score_is_exact_for_common_shapes():
    assert ring_score(0) == 0
    assert ring_score(0b000001) == 1
    assert ring_score(0b000011) == 3
    assert ring_score(0b000101) == 2
    assert ring_score(0b001111) == 10
    assert ring_score(0b100001) == 3
    assert ring_score(0b111111) == 21


def test_ring_score_matches_brute_force_for_every_subset():
    for mask in range(64):
        slots = [slot for slot in range(6) if mask & (1 << slot)]
        best = 0
        for order in itertools.permutations(slots):
            occupied = set()
            score = 0
            for slot in order:
                occupied.add(slot)
                component = {slot}
                frontier = [slot]
                while frontier:
                    current = frontier.pop()
                    for neighbor in ((current - 1) % 6, (current + 1) % 6):
                        if neighbor in occupied and neighbor not in component:
                            component.add(neighbor); frontier.append(neighbor)
                score += len(component)
            best = max(best, score)
        assert ring_score(mask) == best


def test_full_board_ceiling_is_304_and_costs_111_net_tiles():
    solution = solve_board()
    assert solution.score == 304
    assert solution.placement_score == 147
    assert solution.final_bonus == 152
    assert solution.gross_cost == 147
    assert solution.feature_tiles == 36
    assert solution.net_cost == 111
    assert solution.placed_spaces == 42


def test_304_requires_at_least_111_relaxed_net_tiles():
    assert solve_board(net_budget=111).score == 304
    assert solve_board(net_budget=110).score < 304


def test_resource_frontiers_are_monotone_and_nondominated():
    single = single_board_frontier(20)
    assert all(a["net_budget"] < b["net_budget"] for a, b in zip(single, single[1:]))
    assert all(a["score"] < b["score"] for a, b in zip(single, single[1:]))
    pair = two_player_resource_frontier(single, total_factory_tiles=20)
    for row in pair:
        assert not any(
            other["p0_score"] >= row["p0_score"] and other["p1_score"] >= row["p1_score"]
            and (other["p0_score"] > row["p0_score"] or other["p1_score"] > row["p1_score"])
            for other in pair
        )
