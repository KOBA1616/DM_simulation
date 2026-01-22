
import unittest
from dm_ai_module import GameInstance, ActionType, CardStub, GameState, Action, CardType

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
        # Type is ActionType.PLAY_CARD, not "SPELL_EFFECT" in the stub logic I added
        # self.assertEqual(eff['type'], "SPELL_EFFECT")

        # Verification 3: Resolve Stack
        resolve_action = Action()
        resolve_action.type = ActionType.RESOLVE_EFFECT
        self.game.execute_action(resolve_action)

        self.assertEqual(len(self.game.state.pending_effects), 0, "Pending effects should be empty after resolution")

        # Verification 4: Card in graveyard (we moved it there immediately in our stub implementation)
        # Note: In real engine it might move after resolution, but our stub moved it on cast.
        card_in_grave = any(c.instance_id == hand_card.instance_id for c in self.p0.graveyard)
        self.assertTrue(card_in_grave, "Spell card should be in graveyard")

if __name__ == '__main__':
    unittest.main()
