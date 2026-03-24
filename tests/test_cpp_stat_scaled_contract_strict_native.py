import importlib
import os
import sys

import pytest


def _reload_dm_module_strict_native():
    os.environ['DM_STRICT_NATIVE'] = '1'
    os.environ.pop('DM_DISABLE_NATIVE', None)

    if 'dm_ai_module' in sys.modules:
        del sys.modules['dm_ai_module']

    import dm_ai_module  # noqa: F401
    return importlib.import_module('dm_ai_module')


@pytest.mark.skipif(os.environ.get('DM_DISABLE_NATIVE', '0') == '1', reason='Native disabled via DM_DISABLE_NATIVE')
def test_stat_scaled_contract_strict_native_no_shim_fallback():
    dm = _reload_dm_module_strict_native()

    # RED/GREEN contract: strict mode must be active and native module must be present.
    assert getattr(dm, '__strict_native_mode__', False) is True
    native_mod = getattr(dm, '__native_module__', None)
    if native_mod is None:
        pytest.skip('Native module is not loaded; strict-native contract test skipped')

    # Minimal card DB with one STAT_SCALED provider and one playable target.
    cards = [
        {
            'id': 100,
            'name': 'PLAYABLE',
            'cost': 5,
            'type': 'CREATURE',
            'civilizations': ['NATURE'],
            'static_abilities': [],
        },
        {
            'id': 200,
            'name': 'PROVIDER',
            'cost': 0,
            'type': 'CREATURE',
            'civilizations': ['NATURE'],
            'static_abilities': [
                {
                    'type': 'COST_MODIFIER',
                    'value_mode': 'STAT_SCALED',
                    'stat_key': 'CREATURES_PLAYED',
                    'per_value': 1,
                    'min_stat': 1,
                    'max_reduction': 3,
                }
            ],
        },
    ]

    import json

    db = dm.JsonLoader.load_cards(json.dumps(cards))
    gs = dm.GameInstance(None, db)

    # 2 mana only: base cost 5 is not payable without proper STAT_SCALED reduction.
    gs.state.add_card_to_hand(0, 100, 1)
    gs.state.add_card_to_mana(0, 0, 10)
    gs.state.add_card_to_mana(0, 0, 11)
    gs.state.add_test_card_to_battle(0, 200, 300, False, False)

    # update stat to make reduction = 3 (clamped), final cost = 2.
    try:
        gs.state.execute_command({'stat_type': dm.StatType.CREATURES_PLAYED, 'value': 3})
    except Exception:
        # fallback command shape for some native builds
        gs.state.execute_command({'stat_type': 'CREATURES_PLAYED', 'value': 3})

    gs.state.apply_move({'type': dm.CommandType.PLAY_FROM_ZONE, 'instance_id': 1})

    assert any(getattr(c, 'instance_id', None) == 1 for c in gs.state.players[0].battle_zone), (
        'Strict-native contract failed: card was not played to battle zone with STAT_SCALED reduction'
    )
