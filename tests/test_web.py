import json

from azul.agents import RandomAgent
from azul.game import Action, AzulGame, decode_action, encode_action
from azul.web import GameSession, STATIC_ROOT, _action_payload, available_opponents


def test_web_static_assets_and_opponent_catalog_exist():
    assert (STATIC_ROOT / "index.html").is_file()
    assert (STATIC_ROOT / "styles.css").is_file()
    assert (STATIC_ROOT / "app.js").is_file()
    ids = {opponent["id"] for opponent in available_opponents()}
    assert {"competitive_hybrid", "score_neural", "heuristic", "random"} <= ids
    recommended = [opponent for opponent in available_opponents() if opponent["recommended"]]
    assert [opponent["id"] for opponent in recommended] == ["competitive_hybrid"]
    score_champions = [opponent for opponent in available_opponents() if opponent["score_champion"]]
    assert [opponent["id"] for opponent in score_champions] == ["score_neural"]
    assert all(len(opponent["description"]) >= 80 for opponent in available_opponents())


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


def test_web_action_payload_preserves_mixed_draft_and_center_colors():
    game = AzulGame(seed=0)
    mixed_factory = next(
        index
        for index, factory in enumerate(game.factories)
        if factory[game.wild] and any(
            count for color, count in enumerate(factory) if color != game.wild
        )
    )
    mixed_color = next(
        color
        for color, count in enumerate(game.factories[mixed_factory])
        if color != game.wild and count
    )
    draft_id = encode_action(
        Action("draft", source=mixed_factory, color=mixed_color)
    )
    draft_payload = _action_payload(game, draft_id)
    assert draft_payload["color"] == game.action_description(draft_id).split()[1]

    game.phase = "place"
    game.players[0].inventory[1] = 1
    game.players[0].inventory[2] = 1
    center_actions = [
        action_id
        for action_id in game.legal_actions()
        if decode_action(action_id).kind == "place_center"
        and decode_action(action_id).slot == 0
    ]
    assert {
        _action_payload(game, action_id)["color"]
        for action_id in center_actions
    } == {"green", "orange"}


def test_web_game_actions_are_logged_with_training_state(tmp_path):
    session = GameSession(log_dir=tmp_path / "log")
    state = session.new_game("random", seed=1234)
    first_action = state["legal_actions"][0]
    session.act(first_action["id"])

    log_files = list((tmp_path / "log").glob("*.jsonl"))
    assert len(log_files) == 1
    records = [json.loads(line) for line in log_files[0].read_text().splitlines()]
    assert records[0]["event"] == "game_start"
    actions = [record for record in records if record["event"] == "action"]
    assert actions[0]["actor"] == "human"
    assert actions[0]["action_id"] == first_action["id"]
    assert actions[0]["action"] == first_action
    assert len(actions[0]["observation"]) == 282
    assert actions[0]["action_id"] in actions[0]["legal_action_ids"]
    assert any(record["actor"] == "ai" for record in actions)


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
