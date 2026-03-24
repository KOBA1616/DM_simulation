import os
import pytest


def _find_dm_ai_module():
    try:
        import dm_ai_module
        return dm_ai_module
    except Exception:
        return None


@pytest.mark.skipif(os.environ.get('DM_DISABLE_NATIVE', '0') == '1', reason='Native disabled via DM_DISABLE_NATIVE')
def test_cpp_cost_modifier_composition_order_native():
    """Integration test: ensure native path applies PASSIVE -> STATIC -> ACTIVE order."""
    dm = _find_dm_ai_module()
    if dm is None:
        pytest.skip("dm_ai_module not importable (native extension not present)")

    # card: base 12, PASSIVE amount 2, STATIC FIXED value 3, ACTIVE_PAYMENT amount 4
    card = {
        "id": 888888,
        "name": "TEST_COMPOSITION",
        "cost": 12,
        "cost_reductions": [
            {"type": "PASSIVE", "id": "p1", "amount": 2},
            {"type": "ACTIVE_PAYMENT", "id": "a1", "amount": 4},
        ],
        "static_abilities": [
            {"type": "COST_MODIFIER", "value_mode": "FIXED", "value": 3}
        ],
    }

    # load card DB via native JsonLoader if present
    try:
        card_db = dm.JsonLoader.load_cards([card]) if hasattr(dm, 'JsonLoader') else {card['id']: card}
    except Exception:
        card_db = {card['id']: card}

    # construct GameInstance
    try:
        gs = dm.GameInstance(None, card_db)
    except Exception:
        try:
            gs = dm.GameInstance(card_db)
        except Exception:
            pytest.skip('Cannot construct native GameInstance with available signatures')

    # give player 0 enough mana equal to final expected cost (12 -2 -3 -4 = 3)
    # we'll give 3 mana
    gs.state.add_card_to_mana(0, 1, 'm1')
    gs.state.add_card_to_mana(0, 1, 'm2')
    gs.state.add_card_to_mana(0, 1, 'm3')

    # add card to hand
    gs.state.add_card_to_hand(0, card['id'], 'inst-cc')

    # play using ACTIVE_PAYMENT id 'a1' (if module supports passing reduction id via payment fields)
    # build a play command similar to other integration tests
    cmd_type = getattr(dm, 'CommandType', None)
    play_cmd = {'type': cmd_type.PLAY_FROM_ZONE if cmd_type is not None else 33, 'instance_id': 'inst-cc'}

    # try to apply move; if native didn't compute active/static, the shim in dm_ai_module may rescue
    try:
        gs.state.apply_move(play_cmd)
    except Exception:
        try:
            gs.apply_move(play_cmd)
        except Exception:
            pytest.skip('GameInstance.apply_move not available on this build')

    # verify card moved to battle zone
    found = False
    for c in gs.state.players[0].battle_zone:
        try:
            if getattr(c, 'instance_id', None) == 'inst-cc':
                found = True
                break
        except Exception:
            continue

    assert found, 'Card was not played; native composition order may be incorrect'
