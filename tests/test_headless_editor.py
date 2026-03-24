import sys
import os
import unittest

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.editor.core.headless_impl import HeadlessEditorModel
from dm_toolkit.gui.editor.data_manager import CardDataManager
from dm_toolkit.domain.simulation import SimulationRunner

class TestHeadless(unittest.TestCase):
    def test_editor_headless(self):
        # 再発防止: print() は CI 出力を汚染するため削除済み。
        model = HeadlessEditorModel()
        manager = CardDataManager(model)

        # Add Card
        card_item = manager.add_new_card()
        self.assertIsNotNone(card_item)

        # Add Effect
        eff_data = manager.create_default_trigger_data()
        eff_item = manager.add_child_item(card_item, "EFFECT", eff_data, "Effect: ON_PLAY")
        self.assertIsNotNone(eff_item)

        # Verify Data
        full_data = manager.get_full_data()
        self.assertEqual(len(full_data), 1)
        self.assertEqual(full_data[0]['name'], "New Card")
        self.assertEqual(len(full_data[0]['effects']), 1)

    def test_simulation_runner(self):
        # 再発防止: print() は CI 出力を汚染するため削除済み。
        class MockCardDB:
            pass

        # Use a scenario that hopefully exists or handle error
        runner = SimulationRunner(MockCardDB(), "scenario_p0_advantage", episodes=1, threads=1, sims=10, evaluator_type="Random")

        # Just check it instantiates. Running might fail if engine not available.
        # But we can verify it doesn't crash on init.
        self.assertIsNotNone(runner)

        # We verify that we can call cancel (without running)
        runner.cancel()
        self.assertTrue(runner.is_cancelled)

if __name__ == '__main__':
    unittest.main()
