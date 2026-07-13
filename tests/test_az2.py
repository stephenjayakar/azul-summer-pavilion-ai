import numpy as np
import pytest
import random
import torch

from azul.az2 import (
    FastBatchedPUCT, FastNode, StructuredPolicyValueNet, flip_values,
    load_az2_checkpoint, save_az2_checkpoint, value_targets,
)
from azul.game import AzulGame, BONUS_OFFSET, OBS_SIZE
from azul.agents import PolicyValueNet


def test_structured_tokens_cover_observation_and_shapes():
    net = StructuredPolicyValueNet(hidden=64, layers=1, heads=4)
    slices = net.observation_slices()
    coverage = np.zeros(OBS_SIZE, dtype=np.int64)
    for start, stop in slices.values():
        coverage[start:stop] += 1
    assert np.all(coverage == 1)
    obs = torch.from_numpy(np.stack([AzulGame(1).observation(), AzulGame(2).observation()]))
    tokens = net.tokenize(obs)
    logits, values = net(obs)
    assert tokens.shape == (2, 38, 64)
    assert logits.shape == (2, 506)
    assert values.shape == (2, 4)
    assert torch.all((values[:, :2] >= 0) & (values[:, :2] <= 1))
    assert torch.all((values[:, 2:] >= -1) & (values[:, 2:] <= 1))


def test_zero_residual_exactly_preserves_frozen_baseline_policy():
    torch.manual_seed(17)
    baseline = PolicyValueNet().eval()
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4).eval()
    net.baseline.load_state_dict(baseline.state_dict())
    obs = torch.from_numpy(np.stack([AzulGame(17).observation(), AzulGame(18).observation()]))
    with torch.inference_mode():
        expected, _ = baseline(obs)
        actual, _ = net(obs)
    assert torch.equal(expected, actual)


def test_vector_value_orientation():
    p0 = value_targets([120, 80], 0)
    p1 = value_targets([120, 80], 1)
    assert np.allclose(flip_values(p0), p1)
    assert np.allclose(flip_values(p1), p0)


def test_az2_checkpoint_round_trip(tmp_path):
    torch.manual_seed(1)
    net = StructuredPolicyValueNet(hidden=64, layers=1, heads=4)
    net.eval()
    path = save_az2_checkpoint(tmp_path / "model.pt", net, None, 0, {"test": True})
    loaded = load_az2_checkpoint(path)
    obs = torch.from_numpy(AzulGame(4).observation()).unsqueeze(0)
    with torch.inference_mode():
        expected = net(obs)
        actual = loaded(obs)
    assert torch.allclose(expected[0], actual[0])
    assert torch.allclose(expected[1], actual[1])


def test_legacy_az2_checkpoint_loads_with_zero_value_shortcut(tmp_path):
    torch.manual_seed(19)
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4).eval()
    state = net.state_dict()
    state.pop("value_linear.weight")
    state.pop("value_linear.bias")
    for key in list(state):
        if key.startswith("value_mlp."):
            state.pop(key)
    path = tmp_path / "legacy_az2.pt"
    torch.save({
        "format": "az2", "model": state, "optimizer": None,
        "iteration": 0, "config": {}, "hidden": 32, "layers": 1,
        "heads": 4, "obs_size": OBS_SIZE, "num_actions": 506,
    }, path)
    loaded = load_az2_checkpoint(path)
    assert torch.count_nonzero(loaded.value_linear.weight) == 0
    assert torch.count_nonzero(loaded.value_linear.bias) == 0
    assert torch.count_nonzero(loaded.value_mlp[-1].weight) == 0
    assert torch.count_nonzero(loaded.value_mlp[-1].bias) == 0


def test_direct_value_shortcut_receives_gradient():
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4)
    obs = torch.from_numpy(AzulGame(20).observation()).unsqueeze(0)
    _, values = net(obs)
    values.sum().backward()
    assert net.value_linear.weight.grad is not None
    assert torch.count_nonzero(net.value_linear.weight.grad) > 0
    assert net.value_mlp[-1].weight.grad is not None
    assert torch.count_nonzero(net.value_mlp[-1].weight.grad) > 0


def test_fast_puct_returns_legal_policy_and_samples_chance():
    torch.manual_seed(2)
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4)
    game = AzulGame(5)
    search = FastBatchedPUCT(net, torch.device("cpu"), simulations=5, seed=9)
    policies, roots = search.search_many([game], add_noise=False)
    policy = policies[0]
    assert np.isclose(policy.sum(), 1.0)
    assert np.all(policy[~game.legal_mask()] == 0)
    assert sum(child.visits for child in roots[0].children.values()) == 5


@pytest.mark.parametrize("simulations", [1, 100, 400])
def test_deep_search_budgets_return_normalized_legal_visits(simulations):
    torch.manual_seed(22)
    net = StructuredPolicyValueNet(hidden=16, layers=1, heads=4)
    game = AzulGame(23)
    policy, roots = FastBatchedPUCT(
        net, torch.device("cpu"), simulations=simulations, seed=24,
    ).search_many([game], add_noise=False)
    assert np.isclose(policy[0].sum(), 1.0)
    assert np.all(policy[0][~game.legal_mask()] == 0)
    assert sum(child.visits for child in roots[0].children.values()) == simulations


def test_vector_backup_orients_retained_and_switched_turns():
    values = np.asarray([0.6, 0.4, 0.4, 1.0], dtype=np.float32)
    retained = FastNode(to_play=0)
    switched = FastNode(to_play=1)
    FastBatchedPUCT._backup([retained, switched], 0, values)
    assert np.allclose(retained.value, values)
    assert np.allclose(switched.value, flip_values(values))


def test_search_does_not_consume_live_game_rng():
    torch.manual_seed(3)
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4)
    game = AzulGame(6)
    before = game.rng.getstate()
    FastBatchedPUCT(net, torch.device("cpu"), simulations=3, seed=10).search_many(
        [game], add_noise=False,
    )
    assert game.rng.getstate() == before


def test_selected_deterministic_subtree_can_be_reused():
    torch.manual_seed(4)
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4)
    game = AzulGame(7)
    search = FastBatchedPUCT(net, torch.device("cpu"), simulations=12, seed=11)
    policies, roots = search.search_many([game], add_noise=False)
    action = int(np.argmax(policies[0]))
    game.step(action)
    advanced = search.advance_roots(roots, [action], [game])
    assert advanced[0] is not None
    visits = advanced[0].visits
    _, reused = search.search_many([game], add_noise=False, roots=advanced)
    assert reused[0] is advanced[0]
    assert reused[0].visits > visits


def test_cached_root_is_recovered_after_an_observed_move():
    torch.manual_seed(25)
    net = StructuredPolicyValueNet(hidden=16, layers=1, heads=4)
    original = AzulGame(26)
    search = FastBatchedPUCT(net, torch.device("cpu"), simulations=100, seed=27)
    _, roots = search.search_many([original], add_noise=False)
    root = roots[0]
    found = None
    for selected_action, selected in root.children.items():
        if not selected.expanded:
            continue
        after_selected = original.clone(); after_selected.step_fast(selected_action)
        for observed_action, descendant in selected.children.items():
            if not descendant.expanded:
                continue
            current = after_selected.clone(); current.step_fast(observed_action)
            found = (selected_action, descendant, current)
            break
        if found is not None:
            break
    assert found is not None
    selected_action, descendant, current = found
    recovered = search.recover_root_after_observed_moves(
        root, selected_action, current,
    )
    assert recovered is descendant


def test_heuristic_teacher_is_scored_only_at_search_root():
    torch.manual_seed(6)
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4)
    game = AzulGame(13)
    root_actions = len(game.legal_actions())
    search = FastBatchedPUCT(
        net, torch.device("cpu"), simulations=12,
        heuristic_prior_weight=0.5, seed=14,
    )
    calls = 0
    original = search.heuristic._score

    def counted_score(position, action):
        nonlocal calls
        calls += 1
        return original(position, action)

    search.heuristic._score = counted_score
    search.search_many([game], add_noise=False)
    assert calls == root_actions


def _state_before_random_round_fill(seed: int = 31) -> AzulGame:
    game = AzulGame(seed)
    rng = random.Random(seed)
    for _ in range(600):
        for action in game.legal_actions():
            before = game.snapshot(); round_before = game.round
            game.step_fast(action)
            crossed = game.round != round_before and not game.done
            game.restore(before)
            if crossed and len(game.legal_actions()) == 1:
                return game
        game.step(rng.choice(game.legal_actions()))
    raise AssertionError("failed to find chance boundary")


def test_round_boundary_is_resampled_as_chance_not_live_rng():
    torch.manual_seed(5)
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4).eval()
    game = _state_before_random_round_fill()
    other = game.clone(); other.rng.seed(999999)
    first = FastBatchedPUCT(net, torch.device("cpu"), simulations=12, seed=77)
    second = FastBatchedPUCT(net, torch.device("cpu"), simulations=12, seed=77)
    p1, _ = first.search_many([game], add_noise=False)
    p2, _ = second.search_many([other], add_noise=False)
    assert np.allclose(p1[0], p2[0])
    assert len(first.chance_outcomes_seen) >= 2


def test_supply_refill_is_resampled_as_a_chance_boundary():
    torch.manual_seed(7)
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4).eval()
    game = AzulGame(41)
    game.phase = "place"
    game.pending_bonus = 1
    game.pending_bonus_player = game.current_player
    game.supply[:] = [1, 0, 0, 0, 0, 0]
    assert game.legal_actions() == [BONUS_OFFSET]
    before_rng = game.rng.getstate()
    search = FastBatchedPUCT(net, torch.device("cpu"), simulations=12, seed=88)
    policy, _ = search.search_many([game], add_noise=False)
    assert policy[0][BONUS_OFFSET] == 1.0
    assert game.rng.getstate() == before_rng
    assert len(search.chance_outcomes_seen) >= 2
