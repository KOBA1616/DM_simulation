import os
import pytest

# Ensure Python fallback shim is used in test environment
os.environ['DM_DISABLE_NATIVE'] = '1'

import dm_ai_module


def _make_card_def():
    # card 100: playable card; card 200: provider of STAT_SCALED static ability
    return {
        100: type('C', (), {
            'cost': 5,
            'static_abilities': []
        }),
        200: type('C', (), {
            'cost': 0,
            'static_abilities': [
                {
                    'type': 'COST_MODIFIER',
                    'value_mode': 'STAT_SCALED',
                    'stat_key': 'CREATURES_PLAYED',
                    'per_value': 1,
                    'min_stat': 1,
                    'max_reduction': 5,
                }
            ]
        })
    }


def test_play_from_hand_respects_stat_scaled_after_stat_update():
    db = _make_card_def()
    gs = dm_ai_module.GameInstance(None, db)

    # setup: one card in hand (instance 1), and 2 mana available
    gs.state.add_card_to_hand(0, 100, 1)
    gs.state.add_card_to_mana(0, 0, 10)
    gs.state.add_card_to_mana(0, 0, 11)

    # place provider card (200) on battle to provide STAT_SCALED modifier
    gs.state.add_test_card_to_battle(0, 200, 300, False, False)

    # without stat update, base cost is 5 and available mana 2 -> cannot play
    gs.state.apply_move({'type': dm_ai_module.CommandType.PLAY_FROM_ZONE, 'instance_id': 1})
    # still in hand
    assert any(getattr(c, 'instance_id', None) == 1 for c in gs.state.players[0].hand)

    # now update stat: set CREATURES_PLAYED = 3 -> provider card reduces cost by 3 -> final cost 2
    # use StatType enum to ensure provider picks up stat correctly
    gs.state.execute_command({'stat_type': dm_ai_module.StatType.CREATURES_PLAYED, 'value': 3})

    # attempt to play again
    gs.state.apply_move({'type': dm_ai_module.CommandType.PLAY_FROM_ZONE, 'instance_id': 1})
    # should have moved to battle zone
    assert any(getattr(c, 'instance_id', None) == 1 for c in gs.state.players[0].battle_zone)


def test_play_from_hand_blocked_when_stats_below_min():
    db = _make_card_def()
    gs = dm_ai_module.GameInstance(None, db)

    gs.state.add_card_to_hand(0, 100, 2)
    gs.state.add_card_to_mana(0, 0, 20)
    gs.state.add_card_to_mana(0, 0, 21)
    # place provider card on battle to provide STAT_SCALED
    gs.state.add_test_card_to_battle(0, 200, 400, False, False)

    # explicitly set CREATURES_PLAYED = 0 (below min_stat)
    gs.state.execute_command({'stat_type': dm_ai_module.StatType.CREATURES_PLAYED, 'value': 0})

    gs.state.apply_move({'type': dm_ai_module.CommandType.PLAY_FROM_ZONE, 'instance_id': 2})
    # still in hand because no reduction applies
    assert any(getattr(c, 'instance_id', None) == 2 for c in gs.state.players[0].hand)
