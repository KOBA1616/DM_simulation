import unittest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.ai.agent.transformer_model import DuelTransformer

class TestDynamicMaskingV2(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.action_dim = 10
        self.reserved_dim = 20
        self.model = DuelTransformer(
            vocab_size=self.vocab_size,
            action_dim=self.action_dim,
            reserved_dim=self.reserved_dim,
            d_model=32,
            nhead=2,
            num_layers=2,
            dim_feedforward=64,
            max_len=20
        )
        self.model.eval()

    def test_initial_inactive_masking(self):
        """Test that dimensions beyond action_dim are masked by default."""
        batch_size = 2
        seq_len = 5
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, _ = self.model(x)

        # Check active range [0, 10) -> Should have finite values
        active_logits = logits[:, :self.action_dim]
        self.assertTrue(torch.isfinite(active_logits).all(), "Active logits should be finite")

        # Check inactive range [10, 20) -> Should be -inf or very small (-1e9)
        inactive_logits = logits[:, self.action_dim:]
        self.assertTrue((inactive_logits <= -1e8).all(), "Inactive logits should be masked")

    def test_legal_action_masking(self):
        """Test that explicit legal_action_mask masks illegal actions."""
        batch_size = 1
        seq_len = 5
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        # Make action 0 and 1 legal, others illegal
        legal_mask = torch.zeros(batch_size, self.reserved_dim, dtype=torch.bool)
        legal_mask[0, 0] = True
        legal_mask[0, 1] = True

        with torch.no_grad():
            logits, _ = self.model(x, legal_action_mask=legal_mask)

        # 0 and 1 should be finite
        self.assertTrue(torch.isfinite(logits[0, 0]), "Action 0 should be legal")
        self.assertTrue(torch.isfinite(logits[0, 1]), "Action 1 should be legal")

        # Others in active range [2, 10) should be masked
        self.assertTrue((logits[0, 2:self.action_dim] <= -1e8).all(), "Illegal actions in active range should be masked")

        # Inactive range [10, 20) should also be masked
        self.assertTrue((logits[0, self.action_dim:] <= -1e8).all(), "Inactive actions should be masked")

    def test_activate_reserved_actions(self):
        """Test expanding the action space."""
        # Expand by 5 -> New dim = 15
        self.model.activate_reserved_actions(5)
        self.assertEqual(self.model.action_dim, 15)

        batch_size = 1
        seq_len = 5
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, _ = self.model(x)

        # Check new range [0, 15) -> Finite
        self.assertTrue(torch.isfinite(logits[:, :15]).all(), "Expanded actions should be active")

        # Check remaining inactive [15, 20) -> Masked
        self.assertTrue((logits[:, 15:] <= -1e8).all(), "Remaining reserved actions should be masked")

        # Test error on over-expansion
        with self.assertRaises(ValueError):
            self.model.activate_reserved_actions(10) # 15 + 10 = 25 > 20

    def test_predict_action(self):
        """Test the helper method predict_action."""
        # Just check it runs and respects valid actions
        x = torch.randint(0, self.vocab_size, (1, 5))
        legal_actions = [0, 2]

        # Force model to output uniform logits (except masked) to test filtering
        # But we can't easily force output without changing weights.
        # However, if we mask everything but 0 and 2, it MUST pick 0 or 2.

        # Run multiple times to ensure we don't pick illegal
        for _ in range(20):
            action, val = self.model.predict_action(x, legal_actions=legal_actions)
            self.assertIn(action, legal_actions)

if __name__ == '__main__':
    unittest.main()
