import pytest
import json

dm = pytest.importorskip("dm_ai_module")


def test_inspect_active_modifiers_type(tmp_path, capsys):
    cards = [
        {"id": 1, "name": "C1", "type": "CREATURE", "cost": 1, "static_abilities": []}
    ]
    p = tmp_path / "cards.json"
    p.write_text(json.dumps(cards))
    db = dm.JsonLoader.load_cards(str(p))
    game = dm.GameInstance(0, db)
    gs = game.state

    # inspect attribute
    am = getattr(gs, 'active_modifiers', None)
    print('ACTIVE_MODIFIERS_TYPE:', type(am))
    # list available attrs
    try:
        print('DIR_ACTIVE_MODS:', dir(am))
    except Exception as e:
        print('DIR_ERROR:', e)
    # attempt to clear if possible
    try:
        if hasattr(am, 'clear'):
            am.clear()
            print('CLEARED_OK')
    except Exception as e:
        print('CLEAR_ERROR:', e)

    # try append a plain dict and catch exceptions
    try:
        am.append({'reduction_amount': 1})
        print('APPEND_OK')
    except Exception as e:
        print('APPEND_ERROR:', type(e), e)

    # ensure test passes but outputs are captured for inspection
    assert hasattr(gs, 'active_modifiers')
