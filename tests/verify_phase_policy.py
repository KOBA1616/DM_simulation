
import unittest
import torch
import torch.nn as nn
from dm_toolkit.ai.agent.transformer_model import DuelTransformer

class TestPhasePolicy(unittest.TestCase):
    def test_phase_specific_heads(self):
        vocab_size = 100
        action_dim = 10
        d_model = 16

        model = DuelTransformer(vocab_size, action_dim, d_model=d_model)

        # Check if new heads exist
        self.assertTrue(hasattr(model, 'mana_head'), "mana_head missing")
        self.assertTrue(hasattr(model, 'attack_head'), "attack_head missing")
        self.assertTrue(hasattr(model, 'main_head'), "main_head missing")

        # Mock heads with constant outputs to verify dispatch
        def make_dummy_head(val):
            m = nn.Linear(d_model, action_dim)
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, val)
            return m

        model.mana_head = make_dummy_head(100.0)
        model.main_head = make_dummy_head(200.0)
        model.attack_head = make_dummy_head(300.0)
        # Also update policy_head alias just in case (though forward uses main_head directly when phase_ids is present)
        model.policy_head = model.main_head

        input_ids = torch.zeros((3, 10), dtype=torch.long) # Batch 3
        # phases: [MANA, MAIN, ATTACK] -> [2, 3, 4]
        # In dm_ai_module.Phase: MANA=2, MAIN=3, ATTACK=4
        phases = torch.tensor([2, 3, 4], dtype=torch.long)

        output, _ = model(input_ids, phase_ids=phases)

        # Check values
        # Index 0 (MANA) -> 100.0
        # Index 1 (MAIN) -> 200.0
        # Index 2 (ATTACK) -> 300.0

        print("Output 0 mean:", output[0].mean().item())
        print("Output 1 mean:", output[1].mean().item())
        print("Output 2 mean:", output[2].mean().item())

        self.assertTrue(torch.allclose(output[0], torch.tensor(100.0)), f"Expected 100.0, got {output[0].mean().item()}")
        self.assertTrue(torch.allclose(output[1], torch.tensor(200.0)), f"Expected 200.0, got {output[1].mean().item()}")
        self.assertTrue(torch.allclose(output[2], torch.tensor(300.0)), f"Expected 300.0, got {output[2].mean().item()}")

        # Verify default behavior (no phase_ids) -> Main Head (200.0)
        output_default, _ = model(input_ids)
        self.assertTrue(torch.allclose(output_default, torch.tensor(200.0)), "Default should use main_head")

if __name__ == '__main__':
    unittest.main()
