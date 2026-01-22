import unittest
import torch
import numpy as np
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_ai_module import GameInstance, ActionType, CardStub
from dm_toolkit.ai.agent.transformer_model import DuelTransformer

class TestInferenceIntegration(unittest.TestCase):
    def setUp(self):
        self.game = GameInstance()
        self.game.start_game()

        # Check if model exists, if not, create a dummy one for testing
        self.model_dir = os.path.join(project_root, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "test_model.pth")

        # Initialize model structure
        self.model = DuelTransformer(
            vocab_size=1000,
            action_dim=600,
            d_model=32, # Small for test
            nhead=2,
            num_layers=2,
            dim_feedforward=64,
            max_len=200
        )

        # Save dummy model if not exists or always to ensure consistency
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': 1,
        }, self.model_path)

    def test_load_and_infer(self):
        # 1. Load Model
        checkpoint = torch.load(self.model_path)
        loaded_model = DuelTransformer(
            vocab_size=1000,
            action_dim=600,
            d_model=32,
            nhead=2,
            num_layers=2,
            dim_feedforward=64,
            max_len=200
        )
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.eval()

        # 2. Prepare Input (Mock Game State Tokenization)
        # In a real scenario, we would use a Tokenizer. Here we mock it.
        # State: List of token IDs.
        mock_state_tokens = [1, 2, 3, 4, 0, 0] # Example tokens
        state_tensor = torch.tensor([mock_state_tokens], dtype=torch.long)
        padding_mask = (state_tensor == 0)

        # 3. Inference
        with torch.no_grad():
            policy_logits, value_pred = loaded_model(state_tensor, padding_mask=padding_mask)

        # 4. Verify Output Shape
        self.assertEqual(policy_logits.shape, (1, 600), "Policy output should be (batch_size, action_dim)")
        self.assertEqual(value_pred.shape, (1, 1), "Value output should be (batch_size, 1)")

        # 5. Decide Action
        action_idx = torch.argmax(policy_logits, dim=1).item()
        self.assertTrue(0 <= action_idx < 600)

        # 6. Map Action Index to Game Action (Mock Mapping)
        # Assuming 0 is PASS, 1-10 is MANA_CHARGE, etc.
        # This mapping should match `SimpleActionDecoder` mentioned in memory/report.
        # Since we don't have the real decoder here, we just verify we got an index.
        print(f"Inference produced Action Index: {action_idx}")

        # If we want to test execution, we can create a dummy action from this index
        # For this test, verifying the pipeline "Model Load -> Inference -> Index" is sufficient
        # to address "Inference Integration" at the basic level requested.

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == '__main__':
    unittest.main()
