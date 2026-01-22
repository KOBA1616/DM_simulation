import unittest
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_ai_module import GameInstance, ActionType, CardStub
from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from training.ai_player import AIPlayer

class TestInferenceIntegration(unittest.TestCase):
    def setUp(self):
        self.game = GameInstance()
        self.game.start_game()

        # Setup a dummy model
        self.model_path = os.path.join(project_root, "models", "test_integration_model.pth")

        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.model = DuelTransformer(
            vocab_size=1000,
            action_dim=600,
            d_model=32,
            nhead=2,
            num_layers=2,
            dim_feedforward=64,
            max_len=200
        )

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': 1,
        }, self.model_path)

        # Config matching the small model
        self.config = {
            'vocab_size': 1000,
            'action_dim': 600,
            'd_model': 32,
            'nhead': 2,
            'num_layers': 2,
            'dim_feedforward': 64,
            'max_len': 200
        }

    def test_ai_player_execution_flow(self):
        """
        Verify that AIPlayer can:
        1. Observe GameState
        2. Generate an Action using the Model
        3. Execute that Action on the GameInstance
        """
        # 1. Initialize AI Player
        player_id = 0
        ai = AIPlayer(self.model_path, device='cpu', config=self.config)

        # 2. Add cards to hand to give options (ActionEncoder maps index 1-10 to Hand 0-9)
        # We need to ensure the model picks a valid action, but since it's random/untrained,
        # it might pick PASS or INVALID.
        # To make this robust, we will mock the model output OR just accept whatever it does
        # and verify the flow completes without error.

        # Let's add cards
        self.game.state.add_card_to_hand(player_id, 100) # Hand[0]
        self.game.state.add_card_to_hand(player_id, 101) # Hand[1]

        # 3. Get Action
        # We can't force the untrained model to pick a specific action easily without mocking.
        # But we can verify the pipeline runs.

        command = ai.get_action(self.game.state, player_id)

        # 4. Verify Command Object
        self.assertIsNotNone(command)
        # ActionType is IntEnum
        self.assertIsInstance(command.type, (int, ActionType))

        print(f"AI Selected Action Type: {command.type}")

        # 5. Execute Action
        # We call execute_action on GameInstance
        # Note: AIPlayer returns GameCommand, GameInstance expects Action-like object.
        # dm_ai_module.py handles GameCommand fields (type, card_id, etc) via getattr

        initial_mana_count = len(self.game.state.players[player_id].mana_zone)
        initial_hand_count = len(self.game.state.players[player_id].hand)

        self.game.execute_action(command)

        # 6. Verify something happened (or didn't if PASS)
        if command.type == ActionType.MANA_CHARGE:
             new_mana_count = len(self.game.state.players[player_id].mana_zone)
             self.assertEqual(new_mana_count, initial_mana_count + 1)
        elif command.type == ActionType.PASS:
             # Nothing changed
             pass

        # The test passes if no exception was raised during the flow

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
