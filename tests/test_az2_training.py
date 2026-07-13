import json

import numpy as np
import torch

from azul.az2 import StructuredPolicyValueNet
from azul.az2_training import (
    ReplayData, collect_search_vs_opponent, load_human_examples, load_replay,
    save_replay, train_epoch,
)
from azul.agents import HeuristicAgent
from azul.az2_value_calibration import value_only_parameters
from azul.game import AzulGame, NUM_ACTIONS


def _write_human_fixture(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    game = AzulGame(91)
    action = game.legal_actions()[0]
    action_event = {
        "event": "action", "sequence": 0, "actor": "human", "player": 0,
        "action_id": action, "legal_action_ids": game.legal_actions(),
        "observation": game.observation().tolist(),
    }
    complete_name = "complete.jsonl"
    (raw / complete_name).write_text("\n".join(map(json.dumps, [
        {"event": "game_start"}, action_event,
        {"event": "game_end", "done": True, "scores": [100, 80]},
    ])), encoding="utf-8")
    partial_name = "partial.jsonl"
    (raw / partial_name).write_text("\n".join(map(json.dumps, [
        {"event": "game_start"}, action_event,
    ])), encoding="utf-8")
    manifest = {
        "raw_directory": "raw",
        "complete_outcome": [{"file": complete_name, "scores": [100, 80]}],
        "strong_partial": [{"file": partial_name}],
        "limited_opening": [],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_human_loader_assigns_values_only_to_complete_games(tmp_path):
    data = load_human_examples(_write_human_fixture(tmp_path))
    assert len(data) == 2
    assert data.value_valid.tolist() == [True, False]
    assert np.isclose(data.values[0, 0], 0.5)
    assert np.isclose(data.values[0, 1], 0.4)
    assert np.all(data.policies.sum(axis=1) == 1)


def test_compressed_replay_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    masks = rng.random((3, NUM_ACTIONS)) > 0.5
    policies = masks.astype(np.float32); policies /= policies.sum(axis=1, keepdims=True)
    data = ReplayData(
        rng.normal(size=(3, 282)).astype(np.float32), masks, policies,
        rng.normal(size=(3, 4)).astype(np.float32),
        np.asarray([True, False, True]), np.asarray([1.0, 0.2, 1.0], np.float32),
    )
    loaded = load_replay(save_replay(tmp_path / "generation_0000.npz", data))
    assert np.array_equal(loaded.masks, data.masks)
    assert np.allclose(loaded.policies, data.policies, atol=5e-4)
    assert np.allclose(loaded.values, data.values)


def test_multi_head_training_step_is_finite():
    game = AzulGame(92); mask = game.legal_mask()
    policy = mask.astype(np.float32) / mask.sum()
    data = ReplayData(
        game.observation()[None], mask[None], policy[None],
        np.asarray([[0.5, 0.45, 0.2, 1.0]], np.float32),
        np.asarray([True]), np.asarray([1.0], np.float32),
    )
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    metrics = train_epoch(net, optimizer, data, torch.device("cpu"), batch_size=1)
    assert all(np.isfinite(value) for value in metrics.values())


def test_policy_head_only_update_preserves_value_exactly():
    game = AzulGame(93); mask = game.legal_mask()
    policy = np.zeros(NUM_ACTIONS, np.float32); policy[game.legal_actions()[-1]] = 1.0
    data = ReplayData(
        game.observation()[None], mask[None], policy[None],
        np.asarray([[0.5, 0.45, 0.2, 1.0]], np.float32),
        np.asarray([True]), np.asarray([1.0], np.float32),
    )
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in net.policy.parameters():
        parameter.requires_grad_(True)
    net.eval()
    obs = torch.from_numpy(data.obs)
    with torch.inference_mode():
        logits_before, value_before = net(obs)
    optimizer = torch.optim.AdamW(net.policy.parameters(), lr=1e-2)
    train_epoch(
        net, optimizer, data, torch.device("cpu"), batch_size=1,
        policy_weight=1.0, value_weight=0.0,
    )
    net.eval()
    with torch.inference_mode():
        logits_after, value_after = net(obs)
    assert not torch.equal(logits_before, logits_after)
    assert torch.equal(value_before, value_after)


def test_value_calibration_update_preserves_policy_exactly():
    game = AzulGame(94); mask = game.legal_mask()
    policy = mask.astype(np.float32) / mask.sum()
    data = ReplayData(
        game.observation()[None], mask[None], policy[None],
        np.asarray([[0.8, 0.2, 0.8, 1.0]], np.float32),
        np.asarray([True]), np.asarray([1.0], np.float32),
    )
    net = StructuredPolicyValueNet(hidden=32, layers=1, heads=4)
    obs = torch.from_numpy(data.obs)
    net.eval()
    with torch.inference_mode():
        logits_before, value_before = net(obs)
    optimizer = torch.optim.AdamW(value_only_parameters(net), lr=1e-2)
    train_epoch(
        net, optimizer, data, torch.device("cpu"), batch_size=1,
        policy_weight=0.0, value_weight=1.0,
    )
    net.eval()
    with torch.inference_mode():
        logits_after, value_after = net(obs)
    assert torch.equal(logits_before, logits_after)
    assert not torch.equal(value_before, value_after)


def test_opponent_search_replay_has_terminal_targets():
    net = StructuredPolicyValueNet(hidden=16, layers=1, heads=4)
    data, metrics = collect_search_vs_opponent(
        net, HeuristicAgent(95), 2, torch.device("cpu"), 96,
        simulations=1, temperature_moves=0,
    )
    assert len(data) > 0
    assert data.value_valid.all()
    assert np.allclose(data.policies.sum(axis=1), 1.0)
    assert metrics["wins"] + metrics["losses"] + metrics["ties"] == 2
