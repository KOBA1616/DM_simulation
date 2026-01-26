
import unittest
import pytest
import dm_ai_module
from dm_ai_module import GameInstance, ActionType, CardStub, GameState, Action, CardType

@pytest.mark.skipif(not getattr(dm_ai_module, 'IS_NATIVE', False), reason="Requires native engine")
class TestSpellAndStack(unittest.TestCase):
    def setUp(self):
        self.game = GameInstance()
        self.game.start_game()
        self.p0 = self.game.state.players[0]
        self.p1 = self.game.state.players[1]

    def test_spell_casting_stub(self):
        # Setup: Add a "Spell" card to hand.
        # Using ID 7 (Ice and Fire) which is a real Spell in data/cards.json
        spell_card_id = 7
        self.game.state.add_card_to_hand(0, spell_card_id)

        # Verify card is in hand
        hand_card = self.p0.hand[-1]
        self.assertEqual(hand_card.card_id, spell_card_id)

        # Action: Play Card (Cast Spell)
        action = Action()
        action.type = ActionType.PLAY_CARD
        action.card_id = spell_card_id
        action.source_instance_id = hand_card.instance_id
        action.target_player = 0

        # Execute
        self.game.execute_action(action)

        # Verification 1: Card removed from hand
        card_in_hand = any(c.instance_id == hand_card.instance_id for c in self.p0.hand)
        self.assertFalse(card_in_hand, "Spell card should be removed from hand")

        # Verification 2: Pending effects populated
        self.assertEqual(len(self.game.state.pending_effects), 1, "Should have 1 pending effect")
        eff = self.game.state.pending_effects[0]
        # Allow attribute access for stub objects
        cid = getattr(eff, 'card_id', -1)
        self.assertEqual(cid, spell_card_id)

        # Verification 3: Resolve Stack
        pass_action = Action()
        pass_action.type = ActionType.PASS
        self.game.execute_action(pass_action)

        self.assertEqual(len(self.game.state.pending_effects), 0, "Pending effects should be empty after resolution")

        # Verification 4: Card in graveyard
        card_in_grave = any(c.instance_id == hand_card.instance_id for c in self.p0.graveyard)
        self.assertTrue(card_in_grave, "Spell card should be in graveyard")

    def test_stack_lifo(self):
        """Verify that pending effects are resolved in LIFO order."""
        # 1. Add two spells to hand
        card_id_A = 7  # Spell A
        card_id_B = 8  # Spell B
        self.game.state.add_card_to_hand(0, card_id_A)
        self.game.state.add_card_to_hand(0, card_id_B)

        hand_card_A = self.p0.hand[-2]
        hand_card_B = self.p0.hand[-1]

        # 2. Play Spell A
        action_A = Action()
        action_A.type = ActionType.PLAY_CARD
        action_A.card_id = card_id_A
        action_A.source_instance_id = hand_card_A.instance_id
        action_A.target_player = 0
        self.game.execute_action(action_A)

        # 3. Play Spell B (Triggered via some mechanism? Or just stacked?)
        # For this test, we simulate adding another effect to the stack
        # as if Spell A triggered Spell B or we are in a chain.
        # Since we can't easily chain in stub, we just Play B.
        action_B = Action()
        action_B.type = ActionType.PLAY_CARD
        action_B.card_id = card_id_B
        action_B.source_instance_id = hand_card_B.instance_id
        action_B.target_player = 0
        self.game.execute_action(action_B)

        # Verify stack has 2 items: [A, B]
        self.assertEqual(len(self.game.state.pending_effects), 2)
        self.assertEqual(getattr(self.game.state.pending_effects[0], 'card_id'), card_id_A)
        self.assertEqual(getattr(self.game.state.pending_effects[1], 'card_id'), card_id_B)

        # 4. Resolve First (Should be B)
        pass_action = Action()
        pass_action.type = ActionType.PASS
        self.game.execute_action(pass_action)

        # Verify stack has 1 item: [A]
        self.assertEqual(len(self.game.state.pending_effects), 1)
        self.assertEqual(getattr(self.game.state.pending_effects[0], 'card_id'), card_id_A)

        # 5. Resolve Second (Should be A)
        self.game.execute_action(pass_action)

        # Verify stack empty
        self.assertEqual(len(self.game.state.pending_effects), 0)

if __name__ == '__main__':
    unittest.main()
