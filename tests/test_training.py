import numpy as np
import torch

from azul.agents import PolicyValueNet
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
