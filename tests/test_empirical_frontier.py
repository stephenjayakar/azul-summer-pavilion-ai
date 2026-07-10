from azul.empirical_frontier import nondominated_score_pairs


def test_nondominated_score_pairs_removes_weakly_worse_results():
    pairs = [(10, 10), (10, 11), (11, 10), (12, 8), (8, 12), (9, 9), (10, 11)]
    assert nondominated_score_pairs(pairs) == [(8, 12), (10, 11), (11, 10), (12, 8)]
