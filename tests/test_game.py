import random

import numpy as np
import pytest

from azul.game import (
    AzulGame, Action, BOARD_CLOCKWISE, BONUS_OFFSET, CENTER_SOURCE, KEEP_FINISH,
    KEEP_OFFSET, NUM_ACTIONS, OBS_SIZE, PASS_ACTION, decode_action, encode_action,
)


def test_snapshot_restore_and_fast_step_match_validated_engine():
    rng = random.Random(20260712)
    validated = AzulGame(811)
    fast = validated.clone()
    for _ in range(600):
        if validated.done:
            break
        action = rng.choice(validated.legal_actions())
        before = fast.snapshot()
        validated.step(action)
        fast.step_fast(action)
        assert fast.snapshot() == validated.snapshot()
        fast.restore(before)
        assert fast.snapshot() == before
        fast.step_fast(action)
    assert validated.done and fast.done
    assert [p.score for p in validated.players] == [p.score for p in fast.players]


# Literal transcription of all normal-board architectural spaces. Cells are
# (color index, zero-based printed cost); center star is index 6.
EXPECTED_FEATURES = [
    ("window", 0, ((2, 4), (2, 5)), 3),
    ("statue", 0, ((2, 0), (2, 1), (5, 2), (5, 3)), 2),
    ("pillar", 0, ((2, 1), (2, 2), (6, 5), (6, 0)), 1),
    ("window", 1, ((5, 4), (5, 5)), 3),
    ("statue", 1, ((5, 0), (5, 1), (4, 2), (4, 3)), 2),
    ("pillar", 1, ((5, 1), (5, 2), (6, 0), (6, 1)), 1),
    ("window", 2, ((4, 4), (4, 5)), 3),
    ("statue", 2, ((4, 0), (4, 1), (3, 2), (3, 3)), 2),
    ("pillar", 2, ((4, 1), (4, 2), (6, 1), (6, 2)), 1),
    ("window", 3, ((3, 4), (3, 5)), 3),
    ("statue", 3, ((3, 0), (3, 1), (1, 2), (1, 3)), 2),
    ("pillar", 3, ((3, 1), (3, 2), (6, 2), (6, 3)), 1),
    ("window", 4, ((1, 4), (1, 5)), 3),
    ("statue", 4, ((1, 0), (1, 1), (0, 2), (0, 3)), 2),
    ("pillar", 4, ((1, 1), (1, 2), (6, 3), (6, 4)), 1),
    ("window", 5, ((0, 4), (0, 5)), 3),
    ("statue", 5, ((0, 0), (0, 1), (2, 2), (2, 3)), 2),
    ("pillar", 5, ((0, 1), (0, 2), (6, 4), (6, 5)), 1),
]


def set_offers(g, factories, center=None):
    for f in g.factories:
        for c in range(6):
            g.bag[c] += f[c]
    for c in range(6):
        g.bag[c] += g.center_pool[c]
    g.factories = [[0] * 6 for _ in range(5)]
    g.center_pool = [0] * 6
    for i, f in enumerate(factories):
        for c, n in enumerate(f):
            g.factories[i][c] = n
            g.bag[c] -= n
    if center:
        for c, n in enumerate(center):
            g.center_pool[c] = n
            g.bag[c] -= n
    g.assert_invariants()


def give(g, player, color, count):
    g.bag[color] -= count
    g.players[player].inventory[color] += count


def occupy(g, player, star, slot, color=None):
    color = star if color is None else color
    g.bag[color] -= 1
    if star == 6:
        g.players[player].center[slot] = color
    else:
        g.players[player].outer[star][slot] = True


def test_initial_setup_and_conservation():
    g = AzulGame(1)
    assert sum(map(sum, g.factories)) == 20
    assert sum(g.supply) == 10
    assert sum(g.bag) == 102
    g.assert_invariants()


def test_action_round_trip_and_observation_shape():
    examples = [0, 35, 36, 251, 252, 467, 468, 474, 475, 481]
    for i in examples:
        assert encode_action(decode_action(i)) == i
    g = AzulGame(2)
    assert g.observation().shape == (OBS_SIZE,) == (282,)
    assert g.legal_mask().shape == (NUM_ACTIONS,)


def test_observation_distinguishes_public_start_token_holder():
    base = AzulGame(22)
    held_by_me = base.clone()
    held_by_opponent = base.clone()
    held_by_me.next_start_player = 0
    held_by_opponent.next_start_player = 1
    assert not np.array_equal(held_by_me.observation(0), held_by_opponent.observation(0))


def test_factory_draft_takes_nonwild_plus_exactly_one_wild():
    g = AzulGame(3)
    set_offers(g, [[2, 1, 1, 0, 0, 0]])  # purple is wild in round 1
    g.step(encode_action(Action("draft", source=0, color=1)))
    assert g.players[0].inventory == [1, 1, 0, 0, 0, 0]
    assert g.center_pool == [1, 0, 1, 0, 0, 0]


def test_all_wild_source_only_allows_one_tile():
    g = AzulGame(4)
    set_offers(g, [[4, 0, 0, 0, 0, 0]])
    action = encode_action(Action("draft", source=0, color=0))
    assert g.legal_actions() == [action]
    g.step(action)
    assert g.players[0].inventory[0] == 1
    assert g.center_pool[0] == 3


def test_first_center_draft_claims_start_and_penalizes_to_floor_one():
    g = AzulGame(5)
    set_offers(g, [], [1, 0, 0, 0, 0, 4])
    g.players[0].score = 3
    g.step(encode_action(Action("draft", source=CENTER_SOURCE, color=5)))
    assert g.next_start_player == 0
    assert g.players[0].score == 1
    assert g.players[0].inventory[5] == 4
    assert g.players[0].inventory[0] == 1


def test_wild_payment_requires_natural_and_places_natural_tile():
    g = AzulGame(6)
    set_offers(g, [])
    g.phase = "place"
    give(g, 0, 1, 2)
    give(g, 0, 0, 3)
    # Cost 4 can use 1 or 2 green naturals, but never zero.
    ids = g._payment_actions(1, 3, 1)
    assert {decode_action(i).natural for i in ids} == {1, 2}
    before_tower = list(g.tower)
    action = encode_action(Action("place_outer", star=1, slot=3, natural=1))
    g.step(action)
    assert g.players[0].outer[1][3]
    assert g.players[0].inventory[1] == 1
    assert g.players[0].inventory[0] == 0
    assert g.tower[0] == before_tower[0] + 3


def test_connected_scoring_counts_entire_component():
    g = AzulGame(7)
    set_offers(g, [])
    g.phase = "place"
    occupy(g, 0, 1, 0)
    occupy(g, 0, 1, 1)
    give(g, 0, 1, 3)
    score = g.players[0].score
    g.step(encode_action(Action("place_outer", star=1, slot=2, natural=3)))
    assert g.players[0].score == score + 3


def test_all_eighteen_features_match_board_artwork():
    assert BOARD_CLOCKWISE == (2, 5, 4, 3, 1, 0)
    assert list(AzulGame._features()) == EXPECTED_FEATURES


@pytest.mark.parametrize("kind,index,cells,reward", EXPECTED_FEATURES)
def test_each_canonical_feature_triggers(kind, index, cells, reward):
    g = AzulGame(800 + index)
    set_offers(g, [])
    g.phase = "place"
    for star, slot in cells[:-1]:
        occupy(g, 0, star, slot, color=slot if star == 6 else None)
    assert (kind, index) not in g.players[0].claimed
    star, slot = cells[-1]
    cost = slot + 1
    if star == 6:
        color = slot
        give(g, 0, color, cost)
        action = Action("place_center", star=6, slot=slot, color=color, natural=cost)
    else:
        give(g, 0, star, cost)
        action = Action("place_outer", star=star, slot=slot, natural=cost)
    g.step(encode_action(action))
    assert (kind, index) in g.players[0].claimed
    assert g.pending_bonus == reward


def test_pass_keep_four_and_discard_penalty():
    g = AzulGame(9)
    set_offers(g, [])
    g.phase = "place"
    give(g, 0, 1, 6)
    g.players[0].score = 10
    g.step(PASS_ACTION)
    for _ in range(4):
        g.step(KEEP_OFFSET + 1)
    assert KEEP_OFFSET + 1 not in g.legal_actions()
    g.step(KEEP_FINISH)
    assert g.players[0].stored[1] == 4
    assert g.players[0].score == 8


def test_end_game_star_and_number_bonuses_and_leftover_loss():
    g = AzulGame(10)
    p = g.players[0]
    for slot in range(6):
        occupy(g, 0, 0, slot)
    # Complete every cost-1 position (outer six + center slot 0).
    for star in range(1, 6):
        occupy(g, 0, star, 0)
    occupy(g, 0, 6, 0, color=1)
    give(g, 0, 2, 2)
    before = p.score
    g._final_score()
    assert p.score == before + 20 + 4 - 2
    breakdown = g.final_score_breakdown[0]
    assert breakdown == {
        "player": 0,
        "score_before_final": before,
        "completed_stars": [{"color": "purple", "points": 20}],
        "center_bonus": 0,
        "number_bonuses": [{"number": 1, "points": 4}],
        "leftover_penalty": 2,
        "final_bonus_total": 24,
        "final_score": before + 22,
    }


def test_random_games_terminate_and_preserve_tiles():
    for seed in range(20):
        g = AzulGame(seed)
        steps = 0
        rng = random.Random(seed)
        while not g.done:
            g.step(rng.choice(g.legal_actions()))
            steps += 1
            assert steps < 500
        g.assert_invariants()
        assert all(p.score >= 1 for p in g.players)


@pytest.mark.parametrize("num_players,factories", [(2, 5), (3, 7), (4, 9)])
def test_official_player_counts_setup_and_complete(num_players, factories):
    g = AzulGame(1200 + num_players, num_players=num_players)
    assert g.num_factories == factories
    assert len(g.factories) == factories
    assert len(g.players) == num_players
    assert sum(map(sum, g.factories)) == factories * 4
    rng = random.Random(num_players)
    steps = 0
    while not g.done:
        g.step(rng.choice(g.legal_actions()))
        steps += 1
        assert steps < 1000
    g.assert_invariants()


def test_neural_observation_explicitly_rejects_non_two_player_state():
    with pytest.raises(ValueError, match="two-player"):
        AzulGame(77, num_players=3).observation()
