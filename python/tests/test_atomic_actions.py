import pytest
import sys
import os

# Assuming dm_ai_module is available via conftest setup
try:
    import dm_ai_module
except ImportError:
    pass

class TestAtomicActions:
    def test_mana_charge(self, game_state, card_db):
        """Test MANA_CHARGE action using direct C++ bindings"""
        # game_state is now the raw C++ object from python/tests/conftest.py
        state = game_state

        player_id = 0
        card_id = 1
        instance_id = 0

        # Manually add card to hand in C++ state
        state.add_card_to_hand(player_id, card_id, instance_id)

        # Verify initial state
        p = state.get_player(player_id)
        assert len(p.hand) == 1
        assert len(p.mana_zone) == 0

        # Execute Action
        action = dm_ai_module.Action()
        action.type = dm_ai_module.ActionType.MANA_CHARGE
        action.source_hand_card_index = 0
        action.player_id = player_id

        # Run logic
        dm_ai_module.GameLogicSystem.resolve_action(state, action, card_db)

        # Verify result
        p = state.get_player(player_id)
        assert len(p.hand) == 0
        assert len(p.mana_zone) == 1
