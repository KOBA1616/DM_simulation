import unittest
import dm_ai_module
from dm_ai_module import GameState, Zone, Player, CardInstance, GameCommand, TransitionCommand, MutateCommand, MutationType, Civilization

class TestGameCommand(unittest.TestCase):
    def setUp(self):
        self.state = GameState(100)
        self.state.setup_test_duel()
        # Add a test card to Hand
        self.player_id = 0
        self.card_id = 1
        self.instance_id = 0
        self.state.add_card_to_hand(self.player_id, self.card_id, self.instance_id)

    def test_transition_command(self):
        # Move from Hand to Mana
        cmd = TransitionCommand(self.instance_id, Zone.HAND, Zone.MANA, self.player_id)

        # Verify initial state
        hand = self.state.players[self.player_id].hand
        mana = self.state.players[self.player_id].mana_zone
        self.assertEqual(len(hand), 1)
        self.assertEqual(len(mana), 0)
        self.assertEqual(hand[0].instance_id, self.instance_id)

        # Execute
        cmd.execute(self.state)

        # Verify post-execute
        hand = self.state.players[self.player_id].hand
        mana = self.state.players[self.player_id].mana_zone
        self.assertEqual(len(hand), 0)
        self.assertEqual(len(mana), 1)
        self.assertEqual(mana[0].instance_id, self.instance_id)

        # Invert (Undo)
        cmd.invert(self.state)

        # Verify post-invert
        hand = self.state.players[self.player_id].hand
        mana = self.state.players[self.player_id].mana_zone
        self.assertEqual(len(hand), 1)
        self.assertEqual(len(mana), 0)
        self.assertEqual(hand[0].instance_id, self.instance_id)

    def test_mutate_command_tap(self):
        # Add card to battle zone for tapping
        battle_iid = 1
        self.state.add_test_card_to_battle(self.player_id, self.card_id, battle_iid, False, False) # Tapped=False

        cmd = MutateCommand(battle_iid, MutationType.TAP)

        # Verify initial
        card = self.state.get_card_instance(battle_iid)
        self.assertFalse(card.is_tapped)

        # Execute
        cmd.execute(self.state)
        card = self.state.get_card_instance(battle_iid)
        self.assertTrue(card.is_tapped)

        # Invert
        cmd.invert(self.state)
        card = self.state.get_card_instance(battle_iid)
        self.assertFalse(card.is_tapped)

if __name__ == '__main__':
    unittest.main()
