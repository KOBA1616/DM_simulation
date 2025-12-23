
import unittest
import torch
import numpy as np
import tempfile
import os
import sys

# Add bin to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dm_toolkit.training.train_simple import Trainer
from dm_toolkit.training.network_v2 import NetworkV2

class TestTransformerTraining(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.tmp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.tmp_dir, "test_data.npz")

        # Create dummy tokens (jagged arrays)
        # 10 samples
        tokens = []
        policies = []
        values = []

        for _ in range(10):
            seq_len = np.random.randint(10, 50)
            t = np.random.randint(1, 100, size=seq_len)
            tokens.append(t)
            policies.append(np.random.rand(10)) # action size 10
            values.append(np.random.rand(1))

        # Save as object array for tokens
        tokens = np.array(tokens, dtype=object)
        policies = np.array(policies)
        values = np.array(values)

        np.savez(self.data_file, tokens=tokens, policies=policies, values=values)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir)

    def test_transformer_initialization(self):
        trainer = Trainer([self.data_file], save_path=os.path.join(self.tmp_dir, "model.pth"))
        self.assertTrue(trainer.use_transformer)
        self.assertIsInstance(trainer.network, NetworkV2)

    def test_training_loop(self):
        trainer = Trainer([self.data_file], save_path=os.path.join(self.tmp_dir, "model.pth"))
        trainer.train(epochs=1, batch_size=2)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "model.pth")))

if __name__ == "__main__":
    unittest.main()
