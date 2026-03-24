import unittest
import os
import sys
from pathlib import Path

try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 再発防止: GameCommand は抽象C++クラスでインスタンス化不可。CommandDef を使用すること。
from dm_ai_module import GameInstance, CommandType, CardStub, CommandDef
# 再発防止: ActionEncoder は CommandEncoder の後方互換エイリアス。CommandEncoder を使用すること。
from dm_toolkit.ai.agent.tokenization import CommandEncoder, StateTokenizer
if torch:
    from training.ai_player import AIPlayer
    from dm_toolkit.ai.agent.transformer_model import DuelTransformer
else:
    AIPlayer = None
    DuelTransformer = None

class TestAIMasking(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("Torch not installed")
        self.game = GameInstance()
        self.encoder = CommandEncoder()

        # Setup dummy model for AIPlayer
        self.model_path = os.path.join(project_root, "models", "test_masking_model.pth")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model = DuelTransformer(
            vocab_size=10000, action_dim=600, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, max_len=200
        )
        torch.save({'model_state_dict': self.model.state_dict(), 'epoch': 1}, self.model_path)

        self.ai = AIPlayer(self.model_path, device='cpu', config={'vocab_size':10000, 'action_dim':600, 'd_model':32, 'nhead':2, 'num_layers':2, 'dim_feedforward':64, 'max_len':200})

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_encode_command(self):
        # 1. Test PASS
        cmd = CommandDef()
        cmd.type = CommandType.PASS
        idx = self.encoder.encode_command(cmd, self.game.state, 0)
        self.assertEqual(idx, 0)

        # 2. Test MANA_CHARGE
        # Add card to hand
        self.game.state.add_card_to_hand(0, 100, 123) # Hand[0], inst=123
        cmd = CommandDef()
        cmd.type = CommandType.MANA_CHARGE
        cmd.instance_id = 123  # 再発防止: source_instance_id は CommandDef に存在しない。instance_id を使用すること。
        idx = self.encoder.encode_command(cmd, self.game.state, 0)
        self.assertEqual(idx, 1) # 1 + 0

        # 3. Test PLAY_FROM_ZONE
        # Add another card
        self.game.state.add_card_to_hand(0, 101, 124) # Hand[1], inst=124
        cmd = CommandDef()
        cmd.type = CommandType.PLAY_FROM_ZONE  # 再発防止: PLAY_CARD は CommandType に存在しない。PLAY_FROM_ZONE を使用すること。
        cmd.instance_id = 124
        idx = self.encoder.encode_command(cmd, self.game.state, 0)
        self.assertEqual(idx, 11 + 1) # 12

        # 4. Test ATTACK_PLAYER
        # Add card to battle zone
        self.game.state.add_test_card_to_battle(0, 102, 125, False, False) # Battle[0], inst=125
        cmd = CommandDef()
        cmd.type = CommandType.ATTACK_PLAYER
        cmd.instance_id = 125
        idx = self.encoder.encode_command(cmd, self.game.state, 0)
        self.assertEqual(idx, 21 + 0) # 21

    def test_ai_masking(self):
        # マスクで 1 カスのみ有効にすることで AI に特定コマンドを強制選択させる
        valid_indices = [5] # Random valid index

        # We don't care about state content for this test, just masking logic
        cmd = self.ai.get_command(self.game.state, 0, valid_indices)

        # Index 5 -> Mana Charge Hand[4] → Hand[4] が必要
        for i in range(10):
            self.game.state.add_card_to_hand(0, 200+i, 1000+i)

        cmd = self.ai.get_command(self.game.state, 0, valid_indices)

        # Index 5 -> MANA_CHARGE Hand[4]
        # 1 -> Hand[0], 5 -> Hand[4]
        self.assertEqual(cmd.type, CommandType.MANA_CHARGE)
        hand_card_4 = self.game.state.players[0].hand[4]
        self.assertEqual(cmd.instance_id, hand_card_4.instance_id)  # 再発防止: source_instance_id 不可。instance_id を使用すること。

    def test_ai_masking_pass(self):
        valid_indices = [0] # PASS
        cmd = self.ai.get_command(self.game.state, 0, valid_indices)
        self.assertEqual(cmd.type, CommandType.PASS)

if __name__ == '__main__':
    unittest.main()
