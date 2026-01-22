import unittest
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_ai_module import GameInstance, ActionType, CardStub, GameCommand
from dm_toolkit.ai.agent.tokenization import ActionEncoder, StateTokenizer
from training.ai_player import AIPlayer
from dm_toolkit.ai.agent.transformer_model import DuelTransformer

class TestAIMasking(unittest.TestCase):
    def setUp(self):
        self.game = GameInstance()
        self.encoder = ActionEncoder()

        # Setup dummy model for AIPlayer
        self.model_path = os.path.join(project_root, "models", "test_masking_model.pth")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model = DuelTransformer(
            vocab_size=1000, action_dim=600, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, max_len=200
        )
        torch.save({'model_state_dict': self.model.state_dict(), 'epoch': 1}, self.model_path)

        self.ai = AIPlayer(self.model_path, device='cpu', config={'vocab_size':1000, 'action_dim':600, 'd_model':32, 'nhead':2, 'num_layers':2, 'dim_feedforward':64, 'max_len':200})

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_encode_action(self):
        # 1. Test PASS
        cmd = GameCommand()
        cmd.type = ActionType.PASS
        idx = self.encoder.encode_action(cmd, self.game.state, 0)
        self.assertEqual(idx, 0)

        # 2. Test MANA_CHARGE
        # Add card to hand
        self.game.state.add_card_to_hand(0, 100, 123) # Hand[0], inst=123
        cmd = GameCommand()
        cmd.type = ActionType.MANA_CHARGE
        cmd.source_instance_id = 123
        idx = self.encoder.encode_action(cmd, self.game.state, 0)
        self.assertEqual(idx, 1) # 1 + 0

        # 3. Test PLAY_CARD
        # Add another card
        self.game.state.add_card_to_hand(0, 101, 124) # Hand[1], inst=124
        cmd = GameCommand()
        cmd.type = ActionType.PLAY_CARD
        cmd.source_instance_id = 124
        idx = self.encoder.encode_action(cmd, self.game.state, 0)
        self.assertEqual(idx, 11 + 1) # 12

        # 4. Test ATTACK_PLAYER
        # Add card to battle zone
        self.game.state.add_test_card_to_battle(0, 102, 125, False, False) # Battle[0], inst=125
        cmd = GameCommand()
        cmd.type = ActionType.ATTACK_PLAYER
        cmd.source_instance_id = 125
        idx = self.encoder.encode_action(cmd, self.game.state, 0)
        self.assertEqual(idx, 21 + 0) # 21

    def test_ai_masking(self):
        # We want to force the AI to pick a specific action by masking everything else
        # The model is random, so logits are random.
        # But if we mask all but one index, argmax MUST pick that one.

        valid_indices = [5] # Random valid index

        # We don't care about state content for this test, just masking logic
        cmd = self.ai.get_action(self.game.state, 0, valid_indices)

        # We need to reverse map the command to check index
        # Or check if get_action returns command corresponding to index 5
        # Index 5 -> Mana Charge Hand[4]

        # Let's check internal logic by mocking or spying?
        # Simpler: decode the command back
        # But for index 5 to be valid decoding, we need Hand[4] to exist!

        # So we must setup state such that index 5 is valid decoding
        for i in range(10):
            self.game.state.add_card_to_hand(0, 200+i, 1000+i)

        cmd = self.ai.get_action(self.game.state, 0, valid_indices)

        # Verify result is index 5
        # Index 5 -> Mana Charge Hand[4] (indices are 1-based offset for Mana Charge 1-10)
        # Wait, MANA_CHARGE range is 1-10.
        # 1 -> Hand[0]
        # 5 -> Hand[4]

        self.assertEqual(cmd.type, ActionType.MANA_CHARGE)
        # Check source instance id matches Hand[4]
        hand_card_4 = self.game.state.players[0].hand[4]
        self.assertEqual(cmd.source_instance_id, hand_card_4.instance_id)

    def test_ai_masking_pass(self):
        valid_indices = [0] # PASS
        cmd = self.ai.get_action(self.game.state, 0, valid_indices)
        self.assertEqual(cmd.type, ActionType.PASS)

if __name__ == '__main__':
    unittest.main()
