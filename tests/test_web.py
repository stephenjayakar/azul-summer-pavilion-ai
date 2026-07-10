from azul.agents import RandomAgent
from azul.game import Action, AzulGame, encode_action
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


def test_architectural_reward_event_and_progress_are_exposed():
    session = GameSession()
    session.game = AzulGame(seed=44)
    session.agent = RandomAgent(2)
    session.opponent_id = "random"
    game = session.game
    game.phase = "place"
    game.current_player = 0
    # Purple costs 5+6 complete a printed window. Keep conservation exact while
    # constructing the state: one tile is already on the board and six are held.
    game.players[0].outer[0][4] = True
    game.players[0].inventory[0] = 6
    game.bag[0] -= 7
    action = encode_action(Action("place_outer", star=0, slot=5, natural=6))
    state = session.act(action)
    assert state["pending_bonus"] == 3
    assert state["reward_events"] == [{
        "player": 0,
        "kind": "window",
        "index": 5,
        "name": "Purple window",
        "requirement": "Fill costs 5 and 6 on the purple star",
        "reward": 3,
    }]
    purple_window = next(
        feature for feature in state["players"][0]["architecture"]
        if feature["kind"] == "window" and feature["index"] == 5
    )
    assert purple_window["complete"]
    assert purple_window["progress"] == purple_window["required"] == 2
