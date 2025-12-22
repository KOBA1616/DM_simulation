import torch
import torch.nn as nn
import numpy as np
import os

class SynergyGraph(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, matrix_path=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Learnable synergy embeddings
        # Each card gets a vector representation for synergy calculation
        self.synergy_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional: Load pre-computed matrix if path provided
        if matrix_path and os.path.exists(matrix_path):
            try:
                # Assuming matrix is [vocab_size, embedding_dim] or similar
                # If it's a full NxN matrix, we might need a different approach (e.g. SVD to get embeddings)
                # For now, let's assume it provides initial embeddings or we warn.
                data = np.load(matrix_path)
                if isinstance(data, np.ndarray) and data.shape == (vocab_size, embedding_dim):
                    self.synergy_embeddings.weight.data.copy_(torch.from_numpy(data))
                    print(f"Loaded synergy embeddings from {matrix_path}")
                else:
                    print(f"Warning: Matrix at {matrix_path} shape mismatch or invalid format. Expected ({vocab_size}, {embedding_dim}). Initializing randomly.")
            except Exception as e:
                print(f"Warning: Failed to load synergy matrix from {matrix_path}: {e}")
        elif matrix_path:
             print(f"Warning: Synergy matrix path {matrix_path} does not exist.")

    def get_bias_for_sequence(self, sequence):
        """
        Calculates pairwise synergy bias for a sequence of tokens.

        Args:
            sequence: [Batch, SeqLen] (Integer Token IDs)

        Returns:
            bias: [Batch, SeqLen, SeqLen]
        """
        B, S = sequence.shape

        # [Batch, SeqLen, EmbDim]
        embs = self.synergy_embeddings(sequence)

        # Calculate pairwise dot product as synergy score
        # [Batch, SeqLen, EmbDim] @ [Batch, EmbDim, SeqLen] -> [Batch, SeqLen, SeqLen]
        bias = torch.bmm(embs, embs.transpose(1, 2))

        # Scale by sqrt(dim) similar to attention
        bias = bias / (self.embedding_dim ** 0.5)

        return bias
