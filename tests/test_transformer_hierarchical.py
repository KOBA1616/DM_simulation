import unittest
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

from dm_toolkit.ai.agent.transformer_model import (
    DuelTransformerWithActionEmbedding,
    compute_loss_hierarchical,
    extend_action_types,
    encode_action_hierarchical
)

class TestTransformerHierarchical(unittest.TestCase):
    def setUp(self):
        if torch is None:
            # We can't test model methods, but we can test util functions
            pass
        else:
            self.vocab_size = 100
            self.num_action_types = 5
            self.max_params = 10
            self.d_model = 32
            self.model = DuelTransformerWithActionEmbedding(
                vocab_size=self.vocab_size,
                num_action_types=self.num_action_types,
                max_params_per_action=self.max_params,
                d_model=self.d_model,
                nhead=2,
                num_layers=2,
                max_len=50
            )

    def test_forward_shape(self):
        if torch is None: self.skipTest("Torch not installed")
        batch_size = 4
        seq_len = 20
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        type_logits, param_logits, value = self.model(x)

        self.assertEqual(type_logits.shape, (batch_size, self.num_action_types))
        self.assertEqual(param_logits.shape, (batch_size, self.num_action_types, self.max_params))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_compute_loss(self):
        if torch is None: self.skipTest("Torch not installed")
        batch_size = 4
        seq_len = 20
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        outputs = self.model(x)

        # Fake targets
        target_actions = torch.tensor([
            [0, 0], # PASS
            [1, 5], # MANA_CHARGE, index 5
            [2, 3], # PLAY, index 3
            [0, 0]
        ])
        target_values = torch.randn(batch_size)

        loss = compute_loss_hierarchical(outputs, target_actions, target_values)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0) # scalar

    def test_extend_action_types(self):
        if torch is None: self.skipTest("Torch not installed")
        old_num_types = self.model.action_type_embedding.num_embeddings
        num_new = 3
        extend_action_types(self.model, num_new)

        self.assertEqual(self.model.action_type_embedding.num_embeddings, old_num_types + num_new)
        self.assertEqual(self.model.action_type_head[-1].out_features, old_num_types + num_new)

        # Verify forward still works
        batch_size = 2
        x = torch.randint(0, self.vocab_size, (batch_size, 10))
        type_logits, param_logits, _ = self.model(x)

        self.assertEqual(type_logits.shape, (batch_size, old_num_types + num_new))
        self.assertEqual(param_logits.shape, (batch_size, old_num_types + num_new, self.max_params))

    def test_encode_action(self):
        # PASS
        enc = encode_action_hierarchical({'type': 'PASS'})
        self.assertEqual(enc, [0, 0])

        # MANA_CHARGE
        enc = encode_action_hierarchical({'type': 'MANA_CHARGE', 'slot_index': 2})
        self.assertEqual(enc, [1, 2])

        # UNKNOWN -> Default 0? Or mapped?
        # The implementation uses .get(a_type, 0)
        enc = encode_action_hierarchical({'type': 'UNKNOWN_TYPE'})
        self.assertEqual(enc, [0, 0])

if __name__ == '__main__':
    unittest.main()
