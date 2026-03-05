# 再発防止: test_make_unmake_move は unmake_move のハッシュ不一致バグ（エンジン未実装）が
#           解消されるまで常に pytest.skip していたため 2026-03-05 削除。
#           unmake_move が実装されたら test_game_integrity.py に TDD テストとして追加すること。
import unittest
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
