
import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from dm_toolkit.training.network_v2 import NetworkV2, LinearAttention

class TestNetworkV2(unittest.TestCase):
    def test_linear_attention_shape(self):
        """Verify Linear Attention maintains tensor shapes."""
        dim = 32
        seq_len = 10
        batch_size = 2

        layer = LinearAttention(dim, heads=4, dim_head=8)
        x = torch.randn(batch_size, seq_len, dim)

        out = layer(x)
        self.assertEqual(out.shape, (batch_size, seq_len, dim))

    def test_linear_attention_masking(self):
        """Verify masking logic in Linear Attention."""
        dim = 16
        seq_len = 5
        batch_size = 2

        layer = LinearAttention(dim, heads=2, dim_head=8)
        x = torch.randn(batch_size, seq_len, dim)

        # Mask: first element valid, rest invalid
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, 0] = True

        out = layer(x, mask)
        self.assertEqual(out.shape, (batch_size, seq_len, dim))
        # We can't easily assert values without knowing exact initialization, but
        # checking it runs without error with mask is the first step.

    def test_network_v2_forward(self):
        """Verify full network forward pass."""
        vocab_size = 100
        action_space = 10
        embedding_dim = 32
        max_seq_len = 20

        net = NetworkV2(
            embedding_dim=embedding_dim,
            depth=2,
            heads=4,
            input_vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            action_space=action_space
        )

        batch_size = 4
        current_seq_len = 15

        # Random token input
        x = torch.randint(0, vocab_size, (batch_size, current_seq_len))

        policy, value = net(x)

        self.assertEqual(policy.shape, (batch_size, action_space))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_network_v2_masking_integration(self):
        """Verify network handles variable length sequences via mask."""
        vocab_size = 50
        action_space = 5
        net = NetworkV2(input_vocab_size=vocab_size, action_space=action_space, max_seq_len=20)

        batch_size = 2
        seq_len = 10
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        mask[0, 5:] = False # Mask half of first sequence

        policy, value = net(x, mask=mask)

        self.assertEqual(policy.shape, (batch_size, action_space))
        self.assertEqual(value.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()
