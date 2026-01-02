
import sys
import os
import unittest

# Add bin path to load dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bin'))

try:
    import dm_ai_module
except ImportError:
    print("Error: dm_ai_module not found. Please build the project first.")
    sys.exit(1)

class TestCommandUndo(unittest.TestCase):
    def setUp(self):
        # Create a game state with 100 cards
        self.state = dm_ai_module.GameState(100)

        # Setup test duel helper (might set up basic zones)
        # Or manual setup
        self.state.setup_test_duel()

        # Ensure card_owner_map is sized (helper method in binding does resize usually)

    def test_shuffle_undo(self):
        """Verify ShuffleCommand restores the original deck order on invert."""
        pid = 0
        deck_ids = [1, 2, 3, 4, 5]
        self.state.set_deck(pid, deck_ids)

        original_deck = self.state.get_zone(pid, dm_ai_module.Zone.DECK)
        self.assertEqual(len(original_deck), 5)

        # Create and Execute Shuffle
        cmd = dm_ai_module.ShuffleCommand(pid)
        self.state.execute_command(cmd)

        shuffled_deck = self.state.get_zone(pid, dm_ai_module.Zone.DECK)
        self.assertEqual(len(shuffled_deck), 5)
        self.assertCountEqual(shuffled_deck, original_deck) # Same elements

        # Note: It's possible (1/120) that shuffle results in same order.
        # But we primarily test that invert restores the EXACT original order.

        cmd.invert(self.state)

        restored_deck = self.state.get_zone(pid, dm_ai_module.Zone.DECK)
        self.assertEqual(restored_deck, original_deck)

    def test_transition_undo(self):
        """Verify TransitionCommand restores the card position on invert."""
        pid = 0
        cid = 10
        iid = 100

        # Add card to Hand
        self.state.add_card_to_hand(pid, cid, iid)

        # Verify setup
        hand = self.state.get_zone(pid, dm_ai_module.Zone.HAND)
        self.assertIn(iid, hand)

        # Create Move to Mana
        cmd = dm_ai_module.TransitionCommand(iid, dm_ai_module.Zone.HAND, dm_ai_module.Zone.MANA, pid)
        self.state.execute_command(cmd)

        # Verify moved
        hand_after = self.state.get_zone(pid, dm_ai_module.Zone.HAND)
        mana_after = self.state.get_zone(pid, dm_ai_module.Zone.MANA)
        self.assertNotIn(iid, hand_after)
        self.assertIn(iid, mana_after)

        # Undo
        cmd.invert(self.state)

        # Verify restored
        hand_restored = self.state.get_zone(pid, dm_ai_module.Zone.HAND)
        mana_restored = self.state.get_zone(pid, dm_ai_module.Zone.MANA)
        self.assertIn(iid, hand_restored)
        self.assertNotIn(iid, mana_restored)

    def test_mutate_undo(self):
        """Verify MutateCommand (Tap) undo."""
        pid = 0
        cid = 20
        iid = 200

        # Add card to Battle
        self.state.add_test_card_to_battle(pid, cid, iid, False, True) # Not tapped

        # Verify setup
        # Access card instance directly to check tap state
        card = self.state.get_card_instance(iid)
        self.assertFalse(card.is_tapped)

        # Command Tap
        cmd = dm_ai_module.MutateCommand(iid, dm_ai_module.MutationType.TAP)
        self.state.execute_command(cmd)

        self.assertTrue(card.is_tapped)

        # Undo
        cmd.invert(self.state)

        self.assertFalse(card.is_tapped)

    def test_declare_reaction_undo_bug(self):
        """Verify DeclareReactionCommand undo (Bug Replication: pending effects not removed)."""
        pass

if __name__ == '__main__':
    unittest.main()
