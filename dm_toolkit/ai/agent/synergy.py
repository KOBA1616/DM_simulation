
import torch
import numpy as np
import os

class SynergyGraph:
    """
    Manages the card compatibility matrix (Synergy Bias) for the Transformer model.
    """
    def __init__(self, vocab_size, embedding_dim=None, matrix_path=None):
        self.vocab_size = vocab_size
        self.synergy_matrix = None

        if matrix_path and os.path.exists(matrix_path):
            try:
                self.synergy_matrix = torch.tensor(np.load(matrix_path), dtype=torch.float32)
            except Exception as e:
                print(f"Failed to load synergy matrix: {e}")

        if self.synergy_matrix is None:
            # Initialize with random small values or identity for now
            # Shape: (Vocab, Vocab)
            # We use a small random init to break symmetry, but close to 0.
            self.synergy_matrix = torch.randn(vocab_size, vocab_size) * 0.1

    def get_bias_for_sequence(self, input_ids):
        """
        Generates the additive attention bias for a batch of sequences.

        Args:
            input_ids: (Batch, SeqLen) - Token IDs

        Returns:
            bias: (Batch, SeqLen, SeqLen) - Additive bias for attention scores.
        """
        # input_ids: (B, S)
        # We want to lookup S[id_i, id_j] for all i, j in the sequence.

        # 1. Expand input_ids to index the matrix
        # indices_x: (B, S, 1) -> (B, S, S) (repeated along dim 2)
        # indices_y: (B, 1, S) -> (B, S, S) (repeated along dim 1)

        B, S = input_ids.shape
        device = input_ids.device

        if self.synergy_matrix.device != device:
            self.synergy_matrix = self.synergy_matrix.to(device)

        # We need to map TokenID to MatrixIndex.
        # Assuming TokenID maps directly to Matrix Index for simplicity now.
        # In reality, TokenID includes markers. Markers might have 0 synergy.
        # We clamp indices to be within vocab_size.

        valid_mask = (input_ids < self.vocab_size)
        safe_ids = input_ids.clone()
        safe_ids[~valid_mask] = 0 # Redirect invalid IDs to 0

        # Gather logic:
        # matrix: (V, V)
        # We want result (B, S, S)
        # result[b, i, j] = matrix[safe_ids[b, i], safe_ids[b, j]]

        # Use simple indexing
        # Reshape matrix to 1D for flat indexing? No, too complex.
        # Use advanced indexing.

        # ids_flat = safe_ids.view(-1) # (B*S)
        # matrix subset is not easily indexable this way for cross product.

        # Method:
        # 1. Embed rows: (B, S, V) = Embedding(safe_ids) using matrix as weights?
        #    matrix is (V, V).
        #    row_emb = F.embedding(safe_ids, self.synergy_matrix) -> (B, S, V)
        #    This retrieves the row for each token. Row i contains synergies of i with all V.
        # 2. Select columns:
        #    We want the columns corresponding to safe_ids.
        #    gather along dim 2.

        row_emb = torch.nn.functional.embedding(safe_ids, self.synergy_matrix) # (B, S, V)

        # We want to gather specific columns at indices safe_ids from row_emb.
        # safe_ids extended: (B, 1, S) expanded to (B, S, S)
        gather_indices = safe_ids.unsqueeze(1).expand(B, S, S)

        bias = torch.gather(row_emb, 2, gather_indices) # (B, S, S)

        # Mask out interactions involving invalid tokens (if any, though we handled via safe_ids 0)
        # Maybe we want 0 bias for markers.
        # Assuming index 0-1000 are markers and have low/no synergy.

        return bias
