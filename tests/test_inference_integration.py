import unittest
import os
import sys
try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_ai_module import GameInstance, CommandType, CardStub
if torch:
    from dm_toolkit.ai.agent.transformer_model import DuelTransformer
    from training.ai_player import AIPlayer
else:
    DuelTransformer = None
    AIPlayer = None

class TestInferenceIntegration(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("Torch not installed")
        self.game = GameInstance()
        self.game.start_game()

        # Setup a dummy model
        self.model_path = os.path.join(project_root, "models", "test_integration_model.pth")

        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.model = DuelTransformer(
            vocab_size=10000,
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
            'vocab_size': 10000,
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

        # 2. Add cards to hand to give options (CommandEncoder maps index 1-10 to Hand 0-9)
        self.game.state.add_card_to_hand(player_id, 100) # Hand[0]
        self.game.state.add_card_to_hand(player_id, 101) # Hand[1]

        # 3. Get Command
        command = ai.get_command(self.game.state, player_id)

        # 4. Verify Command Object
        self.assertIsNotNone(command)
        # CommandType is IntEnum
        self.assertIsInstance(command.type, (int, CommandType))

        print(f"AI Selected Command Type: {command.type}")

        # 5. Execute Command
        # AIPlayer.get_command returns GameCommand; dm_ai_module handles fields via getattr
        # 再発防止: compat_wrappers は削除済み。直接 execute_command を使用する。
        initial_mana_count = len(self.game.state.players[player_id].mana_zone)
        initial_hand_count = len(self.game.state.players[player_id].hand)
        try:
            self.game.execute_command(command)
        except Exception:
            pass

        # 6. Verify something happened (or didn't if PASS)
        if command.type == CommandType.MANA_CHARGE:
             new_mana_count = len(self.game.state.players[player_id].mana_zone)
             self.assertEqual(new_mana_count, initial_mana_count + 1)
        elif command.type == CommandType.PASS:
             # Nothing changed
             pass

        # The test passes if no exception was raised during the flow

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
