import unittest
import sys
import os
import dm_ai_module

class TestSnapshotLogic(unittest.TestCase):
    def setUp(self):
        # Ensure we can load cards. Assuming running from repo root.
        if os.path.exists("data/cards.json"):
            self.card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        else:
            # Fallback for testing if data not found (mock?)
            # But the module needs real data usually.
            print("Warning: data/cards.json not found.")

        self.game = dm_ai_module.GameInstance(42)
        self.state = self.game.state
        self.game.start_game()

    def test_snapshot_restore(self):
        """Test create_snapshot and restore_snapshot."""
        # Baseline
        initial_hash = self.state.calculate_hash()
        snap = self.state.create_snapshot()

        self.assertEqual(initial_hash, snap.hash_at_snapshot)

        # Modify state using a tracked method (add_card_to_mana uses AddCardCommand)
        # add_test_card_to_battle bypasses command history, so undo() won't work.
        self.state.add_card_to_mana(0, 4000, 100)

        modified_hash = self.state.calculate_hash()
        self.assertNotEqual(initial_hash, modified_hash)

        # Restore
        self.state.restore_snapshot(snap)
        restored_hash = self.state.calculate_hash()

        self.assertEqual(initial_hash, restored_hash)

    def test_make_unmake_move(self):
        """Test make_move and unmake_move with PASS action."""
        initial_hash = self.state.calculate_hash()

        # Create PASS command
        cmd = dm_ai_module.CommandDef()
        cmd.type = dm_ai_module.CommandType.PASS
        # PASS doesn't need targets

        # Execute
        # Note: make_move requires the GameState to have access to CardRegistry (initialized by JsonLoader)
        self.state.make_move(cmd)

        # Check if state changed (hash or history size)
        after_move_hash = self.state.calculate_hash()

        # Pass might imply phase change or at least command history change
        # If Pass is illegal or does nothing, hash might be same?
        # But command history DEFINITELY changes (Action processed).
        # Our hash calculation includes everything?
        # GameState::calculate_hash implementation usually hashes most things.

        # If hash is same, check history size?
        # But we can't easily access history size from Python except strictly via command_history length if exposed as list?
        # command_history is exposed as readonly.
        history_len = len(self.state.command_history)

        self.assertTrue(history_len > 0, "History should not be empty after move")

        # Unmake
        self.state.unmake_move()
        restored_hash = self.state.calculate_hash()

        self.assertEqual(initial_hash, restored_hash)

    def test_nested_snapshots(self):
        """Test multiple snapshots."""
        h0 = self.state.calculate_hash()
        s0 = self.state.create_snapshot()

        # Move 1
        self.state.add_card_to_mana(0, 4001, 101)
        h1 = self.state.calculate_hash()
        s1 = self.state.create_snapshot()

        self.assertNotEqual(h0, h1)

        # Move 2
        self.state.add_card_to_mana(0, 4002, 102)
        h2 = self.state.calculate_hash()

        self.assertNotEqual(h1, h2)

        # Restore to s1
        self.state.restore_snapshot(s1)
        self.assertEqual(self.state.calculate_hash(), h1)

        # Restore to s0
        self.state.restore_snapshot(s0)
        self.assertEqual(self.state.calculate_hash(), h0)

if __name__ == '__main__':
    unittest.main()
