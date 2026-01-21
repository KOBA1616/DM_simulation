import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import dm_ai_module
except ImportError:
    import pytest
    pytest.skip("dm_ai_module not found", allow_module_level=True)

class TestSpellAndStack(unittest.TestCase):
    def setUp(self):
        self.game = dm_ai_module.GameInstance()
        # Mock card DB with a spell
        self.game.card_db = {
            7: {"id": 7, "name": "Test Spell", "type": "SPELL", "card_type": dm_ai_module.CardType.SPELL},
            1: {"id": 1, "name": "Test Creature", "type": "CREATURE", "card_type": dm_ai_module.CardType.CREATURE}
        }
        self.game.state.players[0].hand = []
        self.game.state.players[0].graveyard = []
        self.game.state.players[0].battle_zone = []
        self.game.state.pending_effects = []

        # Add spell to hand
        self.spell_card = dm_ai_module.CardStub(7, 100)
        self.game.state.players[0].hand.append(self.spell_card)

    def test_play_spell_adds_to_stack(self):
        # Action: Play spell
        action = dm_ai_module.Action()
        action.type = dm_ai_module.ActionType.PLAY_CARD
        action.source_instance_id = 100

        self.game.execute_action(action)

        # Verify:
        # 1. Card removed from hand
        self.assertEqual(len(self.game.state.players[0].hand), 0)
        # 2. Card in pending_effects (Stack)
        self.assertEqual(len(self.game.state.pending_effects), 1)
        self.assertEqual(self.game.state.pending_effects[0].card_id, 7)
        # 3. Not in graveyard yet (wait for resolution)
        self.assertEqual(len(self.game.state.players[0].graveyard), 0)

    def test_resolve_effect_moves_to_grave(self):
        # Setup: Spell on stack
        spell_card = dm_ai_module.CardStub(7, 100)
        self.game.state.pending_effects.append(spell_card)

        # Action: Resolve effect
        action = dm_ai_module.Action()
        action.type = dm_ai_module.ActionType.RESOLVE_EFFECT

        self.game.execute_action(action)

        # Verify:
        # 1. Stack empty
        self.assertEqual(len(self.game.state.pending_effects), 0)
        # 2. Card in graveyard
        self.assertEqual(len(self.game.state.players[0].graveyard), 1)
        self.assertEqual(self.game.state.players[0].graveyard[0].card_id, 7)

    def test_play_creature_to_battle_zone(self):
        # Add creature to hand
        creature_card = dm_ai_module.CardStub(1, 200)
        self.game.state.players[0].hand.append(creature_card)

        # Action: Play creature
        action = dm_ai_module.Action()
        action.type = dm_ai_module.ActionType.PLAY_CARD
        action.source_instance_id = 200

        self.game.execute_action(action)

        # Verify:
        # 1. Card removed from hand
        in_hand = any(c.instance_id == 200 for c in self.game.state.players[0].hand)
        self.assertFalse(in_hand)
        # 2. Card in battle zone
        in_bz = any(c.instance_id == 200 for c in self.game.state.players[0].battle_zone)
        self.assertTrue(in_bz)
        # 3. Stack empty
        self.assertEqual(len(self.game.state.pending_effects), 0)

if __name__ == '__main__':
    unittest.main()
