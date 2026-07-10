from azul.web import GameSession, STATIC_ROOT, available_opponents


def test_web_static_assets_and_opponent_catalog_exist():
    assert (STATIC_ROOT / "index.html").is_file()
    assert (STATIC_ROOT / "styles.css").is_file()
    assert (STATIC_ROOT / "app.js").is_file()
    ids = {opponent["id"] for opponent in available_opponents()}
    assert {"competitive_hybrid", "score_neural", "heuristic", "random"} <= ids


def test_random_web_game_advances_back_to_human():
    session = GameSession()
    state = session.new_game("random", seed=1234)
    assert state["started"]
    assert state["current_player"] == 0
    assert state["legal_actions"]
    state = session.act(state["legal_actions"][0]["id"])
    assert state["done"] or state["current_player"] == 0
    assert state["last_ai_actions"]


def test_web_session_rejects_unknown_opponent():
    session = GameSession()
    try:
        session.new_game("not-real")
    except ValueError as exc:
        assert "Unknown opponent" in str(exc)
    else:
        raise AssertionError("unknown opponent should fail")
