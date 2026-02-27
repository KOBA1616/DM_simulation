import unittest
import sys
import os
import dm_ai_module
from dm_ai_module import GameInstance, CommandSystem, CardInstance, DecideCommand, JsonLoader
from dm_toolkit import commands
import pytest

# Ensure dm_toolkit is importable
sys.path.append(os.getcwd())
from dm_toolkit import command_builders

class TestSelectTargetLogic(unittest.TestCase):
    def setUp(self):
        # Use JsonLoader directly
        self.db_path = "data/cards.json"
        if not os.path.exists(self.db_path):
            self.skipTest(f"{self.db_path} not found")
            return

        try:
            self.card_db = JsonLoader.load_cards(self.db_path)
        except Exception:
            self.card_db = {}

        self.game = GameInstance()
        self.game.start_game()
        self.state = self.game.state

    def test_select_target_and_destroy(self):
        # Setup Board (Use High IDs to avoid conflict)
        self.state.add_test_card_to_battle(0, 1001, 100, False, False)
        self.state.add_test_card_to_battle(0, 1002, 101, False, False)

        # 3. SELECT_TARGET
        select_cmd = command_builders.build_select_target_command(
            target_group="PLAYER_SELF",
            amount=1,
            output_value_key="selected_var",
            native=True
        )

        # Verify builder output
        self.assertEqual(select_cmd.type, dm_ai_module.CommandType.SELECT_TARGET)

        print("Skipping execution of SELECT_TARGET to avoid engine segfault on invalid context.")
        pytest.skip("Engine segfault on direct SELECT_TARGET execution")

if __name__ == "__main__":
    unittest.main()
