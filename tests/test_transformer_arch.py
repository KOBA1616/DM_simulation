
import sys
import os
import unittest
import torch
import torch.nn as nn

# Ensure bin and project root are in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dm_toolkit.training.network_v2 import NetworkV2, LinearAttention

class TestTransformerArch(unittest.TestCase):
    def test_network_v2_instantiation(self):
        """
        Test that NetworkV2 can be instantiated with default and custom parameters.
        """
        # Default params
        model = NetworkV2()
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(len(model.layers), 6) # Default depth

        # Custom params
        vocab_size = 500
        action_space = 100
        embedding_dim = 128
        depth = 4
        heads = 4

        model_custom = NetworkV2(
            embedding_dim=embedding_dim,
            depth=depth,
            heads=heads,
            input_vocab_size=vocab_size,
            max_seq_len=100,
            action_space=action_space
        )

        self.assertEqual(len(model_custom.layers), depth)
        self.assertEqual(model_custom.policy_head.out_features, action_space)
        self.assertEqual(model_custom.card_embedding.num_embeddings, vocab_size)

    def test_linear_attention_module(self):
        """
        Verify the structure of the LinearAttention module.
        """
        dim = 64
        heads = 4
        layer = LinearAttention(dim, heads=heads)
        self.assertIsInstance(layer, nn.Module)

        # Basic shape check for qkv projection
        # Input dim -> inner_dim * 3
        # inner_dim = dim_head * heads = 64 * 4 = 256
        # Expected output features = 256 * 3 = 768
        self.assertEqual(layer.to_qkv.in_features, dim)
        self.assertEqual(layer.to_qkv.out_features, 64 * heads * 3)

    def test_forward_pass(self):
        """
        Test a forward pass with dummy data to verify output shapes.
        """
        vocab_size = 100
        action_space = 10
        batch_size = 2
        seq_len = 20

        model = NetworkV2(input_vocab_size=vocab_size, action_space=action_space)

        # Create dummy input tokens [B, SeqLen]
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create dummy mask [B, SeqLen] (optional, but good to test)
        mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        # Mask out last 5 tokens for second sample
        mask[1, -5:] = False

        policy, value = model(x, mask=mask)

        # Check shapes
        # Policy: [B, ActionSpace]
        self.assertEqual(policy.shape, (batch_size, action_space))
        # Value: [B, 1]
        self.assertEqual(value.shape, (batch_size, 1))

        # Check value range (tanh -> -1 to 1)
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))

        # Check policy probability sum (should sum to 1 due to softmax)
        self.assertTrue(torch.allclose(policy.sum(dim=1), torch.ones(batch_size), atol=1e-5))

if __name__ == '__main__':
    unittest.main()
