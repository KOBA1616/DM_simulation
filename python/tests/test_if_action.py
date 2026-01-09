
import sys
import os

# Add bin path for dm_ai_module (relative to python/tests/ -> ../../bin)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bin')))
try:
    import dm_ai_module
except ImportError:
    # Try local build path (relative to python/tests/ -> ../../build)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../build')))
    import dm_ai_module

import unittest

class TestIfAction(unittest.TestCase):
    def setUp(self):
        self.state = dm_ai_module.GameState(100)
        # Use set_deck to clear/init deck
        self.state.set_deck(0, [])
        # Add some cards to deck/hand to manipulate
        for i in range(5):
             self.state.add_card_to_deck(0, 1, i)
             self.state.add_card_to_hand(0, 1, i+10)

    def test_if_input_match_true(self):
        # Setup context
        ctx = {"test_var": 5}

        # Create IF Command
        cmd = dm_ai_module.CommandDef()
        cmd.type = dm_ai_module.CommandType.IF
        cmd.input_value_key = "test_var"

        cond = dm_ai_module.ConditionDef()
        cond.type = "INPUT_VALUE_MATCH"
        cond.value = 5
        cmd.condition = cond

        # If True: Draw 1 card
        true_cmd = dm_ai_module.CommandDef()
        true_cmd.type = dm_ai_module.CommandType.DRAW_CARD
        true_cmd.target_group = dm_ai_module.TargetScope.PLAYER_SELF
        true_cmd.amount = 1
        cmd.if_true = [true_cmd]

        # If False: Discard 1 card
        false_cmd = dm_ai_module.CommandDef()
        false_cmd.type = dm_ai_module.CommandType.DISCARD
        false_cmd.target_group = dm_ai_module.TargetScope.PLAYER_SELF
        false_cmd.amount = 1
        cmd.if_false = [false_cmd]

        # Execute
        initial_hand_size = len(self.state.players[0].hand)
        # Create initial context map

        dm_ai_module.CommandSystem.execute_command(self.state, cmd, -1, 0, ctx)

        # Verify: Hand size should increase by 1 (Draw)
        self.assertEqual(len(self.state.players[0].hand), initial_hand_size + 1)

    def test_if_input_match_false(self):
        # Setup context
        ctx = {"test_var": 3}

        # Create IF Command
        cmd = dm_ai_module.CommandDef()
        cmd.type = dm_ai_module.CommandType.IF
        cmd.input_value_key = "test_var"

        cond = dm_ai_module.ConditionDef()
        cond.type = "INPUT_VALUE_MATCH"
        cond.value = 5
        cmd.condition = cond

        # If True: Draw 1 card
        true_cmd = dm_ai_module.CommandDef()
        true_cmd.type = dm_ai_module.CommandType.DRAW_CARD
        true_cmd.target_group = dm_ai_module.TargetScope.PLAYER_SELF
        true_cmd.amount = 1
        cmd.if_true = [true_cmd]

        # If False: Draw 2 cards (to distinguish from doing nothing)
        false_cmd = dm_ai_module.CommandDef()
        false_cmd.type = dm_ai_module.CommandType.DRAW_CARD
        false_cmd.target_group = dm_ai_module.TargetScope.PLAYER_SELF
        false_cmd.amount = 2
        cmd.if_false = [false_cmd]

        # Execute
        initial_hand_size = len(self.state.players[0].hand)
        dm_ai_module.CommandSystem.execute_command(self.state, cmd, -1, 0, ctx)

        # Verify: Hand size should increase by 2 (False branch)
        self.assertEqual(len(self.state.players[0].hand), initial_hand_size + 2)

if __name__ == '__main__':
    unittest.main()
