from __future__ import annotations

import pytest
import os
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in __import__('sys').path:
    __import__('sys').path.insert(0, _PROJECT_ROOT)

_dm = pytest.importorskip("dm_ai_module", reason="Requires native engine")

pytestmark = pytest.mark.skipif(not bool(getattr(_dm, "IS_NATIVE", False)), reason="Requires native module")


def _make_game():
    db = _dm.CardDatabase()
    # Ensure minimal card definition exists for test card id=1 so target filtering works
    db[1] = _dm.CardDefinition(1, "TestCard", "NONE", [], 0, 1000, _dm.CardKeywords(), [])
    game = _dm.GameInstance(0, db)
    game.state.set_deck(0, [1] * 40)
    game.state.set_deck(1, [1] * 40)
    game.start_game()
    # advance to ATTACK phase
    for _ in range(10):
        if "ATTACK" in str(game.state.current_phase).upper():
            break
        _dm.PhaseManager.next_phase(game.state, db)
    return game, db


def test_untapped_creature_cannot_be_attacked_normally():
    game, db = _make_game()

    # Add attacker (player 0) and untapped target (player 1)
    attacker_iid = 9001
    target_iid = 9002
    game.state.add_test_card_to_battle(0, 1, attacker_iid, False, False)  # untapped, no sickness
    game.state.add_test_card_to_battle(1, 1, target_iid, False, False)    # opponent untapped

    legal = _dm.IntentGenerator.generate_legal_commands(game.state, db)
    # Ensure no ATTACK_CREATURE targeting the untapped target exists
    found = False
    for c in legal:
        if str(c.type).endswith("ATTACK_CREATURE") and getattr(c, "target_instance", None) == target_iid:
            found = True
            break
    assert not found, "Untapped opponent creature should not be attackable by default"


def test_allow_attack_untapped_effect_enables_attack():
    game, db = _make_game()

    attacker_iid = 9011
    target_iid = 9012
    game.state.add_test_card_to_battle(0, 1, attacker_iid, False, False)
    game.state.add_test_card_to_battle(1, 1, target_iid, False, False)

    # Create passive effect allowing this attacker to attack untapped
    p = _dm.PassiveEffect()
    p.type = _dm.PassiveType.ALLOW_ATTACK_UNTAPPED
    p.specific_targets = [attacker_iid]
    p.controller = 0
    game.state.add_passive_effect(p)

    # Debug: inspect passive effects added
    print('passive_count=', game.state.get_passive_effect_count())
    for i, eff in enumerate(game.state.passive_effects):
        print('eff', i, eff.type, getattr(eff, 'specific_targets', None))

    # Use native debug helpers to inspect intermediate decisions
    try:
        allows = _dm.debug_allows_attack_untapped(game.state, attacker_iid, db)
        print('debug_allows_attack_untapped ->', allows)
    except Exception as e:
        print('debug_allows_attack_untapped raised', e)

    try:
        forbidden = _dm.debug_is_attack_forbidden(game.state, attacker_iid, target_iid, db)
        print('debug_is_attack_forbidden ->', forbidden)
    except Exception as e:
        print('debug_is_attack_forbidden raised', e)

    legal = _dm.IntentGenerator.generate_legal_commands(game.state, db)
    found = False
    for c in legal:
        if str(c.type).endswith("ATTACK_CREATURE") and getattr(c, "target_instance", None) == target_iid:
            found = True
            break
    assert found, "ALLOW_ATTACK_UNTAPPED passive should enable attacking untapped creatures"
