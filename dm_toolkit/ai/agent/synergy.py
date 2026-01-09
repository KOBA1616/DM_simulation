import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Optional, cast, Any, List, Dict

class SynergyGraph(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, matrix_path: Optional[str] = None, use_dense_matrix: bool = False) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim
        self.use_dense_matrix = use_dense_matrix

        if self.use_dense_matrix:
            # Dense matrix: [Vocab, Vocab]
            # Register as buffer to be part of state_dict but not a parameter
            self.register_buffer("synergy_matrix", torch.zeros(vocab_size, vocab_size))
        else:
            # Learnable synergy embeddings
            # Each card gets a vector representation for synergy calculation
            self.synergy_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional: Load pre-computed matrix if path provided
        if matrix_path and os.path.exists(matrix_path) and not self.use_dense_matrix:
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
        elif matrix_path and not self.use_dense_matrix:
             print(f"Warning: Synergy matrix path {matrix_path} does not exist.")

    @classmethod
    def from_manual_pairs(cls, vocab_size: int, json_path: str, device: str = 'cpu') -> 'SynergyGraph':
        """
        Creates a SynergyGraph initialized with manual pairs from a JSON file.
        Uses a dense matrix representation.
        """
        model = cls(vocab_size, use_dense_matrix=True)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                pairs = json.load(f)

            # Populate matrix
            matrix = torch.zeros(vocab_size, vocab_size)
            count = 0
            for p in pairs:
                c1 = p.get('card_id_1')
                c2 = p.get('card_id_2')
                score = p.get('synergy_score', 0.0)
                if c1 is not None and c2 is not None:
                    # Check bounds
                    if c1 < vocab_size and c2 < vocab_size:
                        matrix[c1, c2] = score
                        matrix[c2, c1] = score # Symmetric
                        count += 1

            model.synergy_matrix.copy_(matrix)
            model.to(device)
            print(f"Loaded {count} synergy pairs from {json_path}")
        else:
            print(f"Warning: Manual pairs file {json_path} not found.")

        return model

    def get_bias_for_sequence(self, sequence: torch.Tensor) -> Any:
        """
        Calculates pairwise synergy bias for a sequence of tokens.

        Args:
            sequence: [Batch, SeqLen] (Integer Token IDs)

        Returns:
            bias: [Batch, SeqLen, SeqLen]
        """
        B, S = sequence.shape

        if self.use_dense_matrix:
            # Look up in dense matrix
            # sequence: [B, S]
            # We want result: [B, S, S]
            # result[b, i, j] = matrix[sequence[b, i], sequence[b, j]]

            # Advanced indexing:
            # row_indices: sequence.unsqueeze(2) -> [B, S, 1]
            # col_indices: sequence.unsqueeze(1) -> [B, 1, S]
            # matrix is [V, V]
            # We need to map batch indices.

            # Using simple gathered lookup might be clearer or broadcasting.
            # But matrix is fixed, not batched.

            # self.synergy_matrix[sequence[:, :, None], sequence[:, None, :]] works if matrix supports it?
            # No, standard indexing [index_tensor, index_tensor] works for first dimensions usually.

            # matrix is [V, V]
            # indices are [B, S, 1] and [B, 1, S]
            # We can treat batch as flattened or use gathering.

            # Efficient implementation:
            # 1. Expand matrix to batch (if needed) or use F.embedding
            # Actually, we can treat the matrix as an embedding table of size [V, V]
            # Each row i contains the synergy of i with all other cards.
            # 1. Embed sequence using the matrix:
            #    row_embeddings = F.embedding(sequence, self.synergy_matrix) -> [B, S, V]
            #    This gives synergy of each card in sequence with ALL cards in vocab.
            # 2. Gather the columns corresponding to the sequence.
            #    We want columns corresponding to sequence indices.
            #    torch.gather(row_embeddings, 2, sequence.unsqueeze(1).expand(-1, S, -1))

            row_embeddings = torch.nn.functional.embedding(sequence, self.synergy_matrix) # [B, S, V]

            # Gather relevant columns
            # target indices: [B, S, S]
            target_indices = sequence.unsqueeze(1).expand(-1, S, -1)
            bias = torch.gather(row_embeddings, 2, target_indices) # [B, S, S]

            return bias

        else:
            # [Batch, SeqLen, EmbDim]
            embs = cast(torch.Tensor, self.synergy_embeddings(sequence))

            # Calculate pairwise dot product as synergy score
            # [Batch, SeqLen, EmbDim] @ [Batch, EmbDim, SeqLen] -> [Batch, SeqLen, SeqLen]
            bias = torch.bmm(embs, embs.transpose(1, 2))

            # Scale by sqrt(dim) similar to attention
            bias = bias / (self.embedding_dim ** 0.5)

            return bias
