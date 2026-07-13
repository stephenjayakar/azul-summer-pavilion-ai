import numpy as np
import torch

from azul.agents import PolicyValueNet
from azul.alphazero import (
    BatchedPUCT, collect_search_self_play, policy_value_update,
    collect_value_bootstrap_games, value_bootstrap_update, arena_evaluate,
)
from azul.game import AzulGame
from azul.training import collect_self_play, multi_star_potential, score_potential


def test_initial_score_potential_is_zero():
    game = AzulGame(1)
    assert score_potential(game.players[0]) == 0.0
    assert multi_star_potential(game.players[0]) == 0.0


def test_score_objective_episode_reward_equals_final_score_delta():
    torch.manual_seed(3)
    net = PolicyValueNet()
    records, _, scores, stats = collect_self_play(
        net, games_count=4, device=torch.device("cpu"), seed=90,
        objective="score", shaping_weight=1.0,
    )
    observed = sum(record["reward"] for record in records)
    expected = (scores.sum() - scores.size * 5) / 20.0
    assert np.isclose(observed, expected, atol=1e-5)
    assert all("return" in record and "adv" in record for record in records)
    assert 0 <= stats["outer_completion_rate"] <= 1


def test_team_score_objective_credits_exact_combined_score_to_each_seat():
    torch.manual_seed(4)
    net = PolicyValueNet()
    records, _, scores, stats = collect_self_play(
        net, games_count=3, device=torch.device("cpu"), seed=190,
        objective="team_score", shaping_weight=0.0,
    )
    # Each game has two player trajectories, and each trajectory receives the
    # exact combined table-score delta from the initial 5+5 points.
    expected = 2 * (scores.sum() - scores.shape[0] * 10) / 20.0
    assert np.isclose(sum(record["reward"] for record in records), expected, atol=1e-5)
    assert np.isclose(stats["mean_combined_score"], scores.sum(axis=1).mean())
    assert stats["max_individual_score"] == scores.max()


def test_puct_returns_normalized_legal_policy():
    torch.manual_seed(5)
    game = AzulGame(22)
    net = PolicyValueNet()
    search = BatchedPUCT(
        net, torch.device("cpu"), simulations=3,
        dirichlet_fraction=0.0, seed=5,
    )
    policy = search.search_many([game], add_noise=False)[0]
    assert np.isclose(policy.sum(), 1.0)
    assert np.all(policy[~game.legal_mask()] == 0)
    assert np.count_nonzero(policy) <= 3


def test_search_self_play_targets_match_final_player_margin():
    torch.manual_seed(6)
    net = PolicyValueNet()
    records, scores, stats = collect_search_self_play(
        net, games_count=1, device=torch.device("cpu"), seed=23,
        simulations=1, dirichlet_fraction=0.0, temperature_moves=0,
        value_scale=50.0,
    )
    assert records
    for record in records:
        player = record["player"]
        expected = np.clip((scores[0, player] - scores[0, 1 - player]) / 50.0, -1, 1)
        assert np.isclose(record["value_target"], expected)
        assert np.isclose(record["policy"].sum(), 1.0)
        assert np.all(record["policy"][~record["mask"]] == 0)
    assert stats["examples"] == len(records)


def test_policy_value_update_accepts_search_targets():
    torch.manual_seed(7)
    net = PolicyValueNet()
    game = AzulGame(24)
    mask = game.legal_mask()
    policy = mask.astype(np.float32) / mask.sum()
    records = [{
        "obs": game.observation(), "mask": mask, "policy": policy,
        "value_target": 0.25,
    }]
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    losses = policy_value_update(
        net, optimizer, records, torch.device("cpu"), epochs=1, batch_size=1,
    )
    assert all(np.isfinite(loss) for loss in losses)


def test_value_bootstrap_preserves_policy_parameters():
    torch.manual_seed(8)
    net = PolicyValueNet()
    records, _ = collect_value_bootstrap_games(
        net, games_count=1, device=torch.device("cpu"), seed=25,
    )
    policy_before = [parameter.detach().clone() for parameter in net.policy.parameters()]
    loss = value_bootstrap_update(
        net, records, torch.device("cpu"), epochs=1, batch_size=512,
    )
    assert np.isfinite(loss)
    assert all(torch.equal(before, after) for before, after in zip(policy_before, net.policy.parameters()))


def test_arena_evaluation_accounts_for_every_game():
    torch.manual_seed(9)
    net = PolicyValueNet()
    result = arena_evaluate(net, net, 2, torch.device("cpu"), seed=26)
    assert result["arena_wins"] + result["arena_losses"] + result["arena_ties"] == 2
    assert 0 <= result["arena_win_rate"] <= 1
