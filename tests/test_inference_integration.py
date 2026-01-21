import sys
import os
import unittest
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.ai_player import AIPlayer
import dm_ai_module

class TestInferenceIntegration(unittest.TestCase):
    def test_inference_flow(self):
        # 1. Setup Game
        game = dm_ai_module.GameInstance()
        game.start_game()

        # 2. Setup AI Player
        # We try to find the latest model in models/
        import glob
        pths = sorted(glob.glob("models/duel_transformer_*.pth"), reverse=True)
        if not pths:
            print("No model found, skipping inference test with warning")
            return

        model_path = pths[0]
        print(f"Testing inference with model: {model_path}")

        ai = AIPlayer(model_path, device='cpu')

        # 3. Get Action
        cmd, action_idx = ai.get_action(game.state)

        # 4. Verify output
        print(f"AI chose action index: {action_idx}")
        self.assertIsInstance(cmd, dm_ai_module.GameCommand)
        self.assertTrue(hasattr(cmd, 'type'))

        # 5. Execute Action (Mock)
        # Since we don't have full mapping in AIPlayer yet, we just check command type is valid enum
        self.assertIsInstance(cmd.type, (int, dm_ai_module.ActionType))

        # Verify model loading happened (private attribute check just for verification assurance)
        self.assertTrue(hasattr(ai, 'model'))
        self.assertIsInstance(ai.model, torch.nn.Module)

if __name__ == '__main__':
    unittest.main()
