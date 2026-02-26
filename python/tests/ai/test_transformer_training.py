import unittest
import os
import shutil
import tempfile
import numpy as np
import torch
from dm_toolkit.training.train_simple import Trainer
from dm_toolkit.ai.agent.transformer_network import NetworkV2

class TestTransformerTraining(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.tmp_dir, "test_data.npz")

        tokens = []
        policies = []
        values = []

        vocab_size = 100
        action_space = 600

        for _ in range(100):
            seq_len = np.random.randint(10, 50)
            t = np.random.randint(0, vocab_size, seq_len).astype(np.int64)
            tokens.append(t)

            p = np.random.rand(action_space).astype(np.float32)
            p /= p.sum()
            policies.append(p)

            v = np.random.rand(1).astype(np.float32)
            values.append(v)

        tokens_arr = np.array(tokens, dtype=object)
        policies_arr = np.array(policies, dtype=np.float32)
        values_arr = np.array(values, dtype=np.float32)

        np.savez(self.data_file, tokens=tokens_arr, policies=policies_arr, values=values_arr)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_transformer_initialization(self):
        trainer = Trainer([self.data_file], save_path=os.path.join(self.tmp_dir, "model.pth"), force_network_type="transformer")
        self.assertTrue(trainer.use_transformer)
        self.assertIsInstance(trainer.network, NetworkV2)

    def test_training_loop(self):
        trainer = Trainer([self.data_file], save_path=os.path.join(self.tmp_dir, "model.pth"), force_network_type="transformer")
        trainer.train(epochs=1, batch_size=10)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "model.pth")))

if __name__ == "__main__":
    unittest.main()
