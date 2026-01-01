
import os
import sys
import numpy as np
import torch
import unittest
import shutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from dm_toolkit.training.train_simple import Trainer

class TestTrainPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = "temp_test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.npz_path = os.path.join(self.test_dir, "test_data.npz")
        self.model_path = os.path.join(self.test_dir, "test_model.pth")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_train_with_transformer_data(self):
        # Generate dummy data for Transformer (tokens)
        num_samples = 10
        seq_len = 20
        action_size = 600
        vocab_size = 50

        # Tokens: list of arrays (simulating jagged arrays from different games)
        tokens = [np.random.randint(0, vocab_size, (seq_len,), dtype=np.int64) for _ in range(num_samples)]

        policies = np.random.rand(num_samples, action_size).astype(np.float32)
        values = np.random.rand(num_samples).astype(np.float32)

        # Save as npz
        np.savez(self.npz_path, tokens=tokens, policies=policies, values=values)

        # Initialize Trainer
        trainer = Trainer([self.npz_path], save_path=self.model_path)

        # Assert mode detection
        self.assertTrue(trainer.use_transformer)
        self.assertIsNotNone(trainer.network)

        # Run 1 epoch
        trainer.train(epochs=1, batch_size=2)

        self.assertTrue(os.path.exists(self.model_path))

    def test_train_with_resnet_data(self):
        # Generate dummy data for ResNet (states)
        num_samples = 10
        state_dim = 100
        action_size = 600

        states = np.random.rand(num_samples, state_dim).astype(np.float32)
        policies = np.random.rand(num_samples, action_size).astype(np.float32)
        values = np.random.rand(num_samples).astype(np.float32)

        np.savez(self.npz_path, states=states, policies=policies, values=values)

        trainer = Trainer([self.npz_path], save_path=self.model_path)

        self.assertFalse(trainer.use_transformer)

        trainer.train(epochs=1, batch_size=2)
        self.assertTrue(os.path.exists(self.model_path))

if __name__ == '__main__':
    unittest.main()
