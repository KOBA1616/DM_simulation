
import os
import sys
import unittest
import torch
import numpy as np
import tempfile
import shutil

# Setup path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
    from dm_toolkit.ai.agent.network import AlphaZeroTransformer, LinearAttention
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class TestTransformerIntegration(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.action_size = 600
        self.vocab_size = 1000
        self.max_seq_len = 50
        self.model = AlphaZeroTransformer(
            action_size=self.action_size,
            embedding_dim=64,
            depth=2,
            heads=4,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len
        ).to(self.device)
        self.model.eval()

    def test_model_forward(self):
        """Test if the model accepts input and produces correct output shapes."""
        batch_size = 4
        seq_len = 20
        dummy_input = torch.randint(0, self.vocab_size, (batch_size, seq_len)).to(self.device)

        # Create mask (all valid for simplicity)
        mask = torch.ones((batch_size, seq_len), dtype=torch.bool).to(self.device)

        policy, value = self.model(dummy_input, mask)

        self.assertEqual(policy.shape, (batch_size, self.action_size))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_model_masking(self):
        """Test if masking works (masked tokens shouldn't affect output ideally, but mainly just run without error)."""
        batch_size = 2
        seq_len = 10
        dummy_input = torch.randint(0, self.vocab_size, (batch_size, seq_len)).to(self.device)
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool).to(self.device)
        mask[:, :5] = True # First 5 tokens valid

        policy, value = self.model(dummy_input, mask)
        self.assertEqual(policy.shape, (batch_size, self.action_size))

    def test_save_load(self):
        """Test saving and loading the model state dict."""
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
            torch.save(self.model.state_dict(), tmp.name)

            new_model = AlphaZeroTransformer(
                action_size=self.action_size,
                embedding_dim=64,
                depth=2,
                heads=4,
                vocab_size=self.vocab_size,
                max_seq_len=self.max_seq_len
            ).to(self.device)

            new_model.load_state_dict(torch.load(tmp.name))
            new_model.eval()

            dummy_input = torch.randint(0, self.vocab_size, (1, 10)).to(self.device)
            p1, v1 = self.model(dummy_input)
            p2, v2 = new_model(dummy_input)

            self.assertTrue(torch.allclose(p1, p2))
            self.assertTrue(torch.allclose(v1, v2))

    def test_cpp_integration_callback(self):
        """Mock the C++ callback behavior using the defined model."""

        def sequence_batch_inference(token_lists):
            # Same logic as in verify_performance.py
            max_len = 0
            for tokens in token_lists:
                max_len = max(max_len, len(tokens))
            max_len = max(max_len, 1)

            batch_size = len(token_lists)
            padded_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
            mask_tensor = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)

            for i, tokens in enumerate(token_lists):
                L = len(tokens)
                if L > 0:
                    t = torch.tensor(tokens, dtype=torch.long, device=self.device)
                    padded_tensor[i, :L] = t
                    mask_tensor[i, :L] = True

            with torch.no_grad():
                policy_logits, values = self.model(padded_tensor, mask=mask_tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                vals = values.squeeze(1).cpu().numpy()
                return policies.tolist(), vals.tolist()

        # Simulate C++ calling this
        batch_input = [
            [1, 2, 3],
            [4, 5, 6, 7, 8],
            [] # Empty sequence case
        ]

        policies, values = sequence_batch_inference(batch_input)

        self.assertEqual(len(policies), 3)
        self.assertEqual(len(values), 3)
        self.assertEqual(len(policies[0]), self.action_size)

if __name__ == '__main__':
    unittest.main()
