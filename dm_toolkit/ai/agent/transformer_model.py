try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    # Dummy classes to allow import without torch
    class torch:
        class Tensor:
            pass
        @staticmethod
        def randn(*args): return torch.Tensor()
    class nn:
        class Module:
            pass
        class Embedding(Module):
            def __init__(self, *args, **kwargs): pass
        class Parameter:
            def __init__(self, *args, **kwargs): pass
        class TransformerEncoderLayer(Module):
            def __init__(self, *args, **kwargs): pass
        class TransformerEncoder(Module):
            def __init__(self, *args, **kwargs): pass
        class Sequential(Module):
            def __init__(self, *args, **kwargs): pass
        class LayerNorm(Module):
            def __init__(self, *args, **kwargs): pass
        class Linear(Module):
            def __init__(self, *args, **kwargs): pass
        class GELU(Module):
            pass
    class F:
        pass

import math
from typing import Optional, Tuple
from dm_toolkit.ai.agent.synergy import SynergyGraph

class DuelTransformer(nn.Module):
    """
    Phase 8 Transformer Architecture (Sequence Understanding).

    Replaces the fixed-length feature vector with a token-based sequence model.
    Incorporates Synergy Bias Mask for card compatibility understanding.

    Specs:
    - Input: Token Sequence (Integer IDs)
    - Architecture: Encoder-Only Transformer
    - d_model: 256
    - Layers: 6
    - Heads: 8
    - d_ff: 1024
    - Activation: GELU
    - Context Length: Dynamic (Max ~512)
    """
    def __init__(self, vocab_size: int, action_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 1024, max_len: int = 200, synergy_matrix_path: Optional[str] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead

        # 1. Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # store positional embeddings without a leading batch dim so we can
        # expand dynamically to the runtime batch size during forward.
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # 2. Synergy Manager (optional for speed)
        self.use_synergy = synergy_matrix_path is not None
        # annotate attribute for gradual typing
        self.synergy_graph: Optional[SynergyGraph] = None
        if self.use_synergy:
            self.synergy_graph = SynergyGraph(vocab_size, matrix_path=synergy_matrix_path)

        # 3. Transformer Encoder
        # We use a custom encoder block loop or standard encoder with custom mask logic.
        # nn.TransformerEncoderLayer allows passing src_mask (additive).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Heads
        # Policy Head: Predicts action logits from the "CLS" token (index 0) or Global Pooling
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim)
        )

        # Value Head: Predicts win probability
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, SeqLen] (Integer Token IDs)
            padding_mask: [Batch, SeqLen] (Boolean, True = Pad/Ignored) - Optional
        """
        B, S = x.shape

        # 1. Embedding
        # [Batch, SeqLen, d_model]
        emb = self.token_embedding(x)

        # Add Positional Embedding (Slice to current seq len)
        seq_len = min(S, self.max_len)
        # make positional embeddings have batch dimension at runtime
        pos = self.pos_embedding[:seq_len, :].unsqueeze(0).expand(B, seq_len, self.d_model)
        emb = emb[:, :seq_len, :] + pos

        # 2. Synergy Bias (optional for speed)
        if self.use_synergy and self.synergy_graph is not None:
            # [Batch, SeqLen, SeqLen]
            # bias[b, i, j] is the value to add to attention score.
            synergy_bias = self.synergy_graph.get_bias_for_sequence(x)

            # PyTorch MultiheadAttention expects mask of shape (Batch * NumHeads, SeqLen, SeqLen)
            # if it's 3D.
            # Our synergy_bias is (Batch, SeqLen, SeqLen).
            # We need to repeat it for each head.
            # Synergy is applied equally to all heads.

            # (Batch, Seq, Seq) -> (Batch, 1, Seq, Seq) -> (Batch, NumHeads, Seq, Seq) -> (Batch*NumHeads, Seq, Seq)
            synergy_bias = synergy_bias.unsqueeze(1).repeat(1, self.nhead, 1, 1)
            # use the actual sequence length used for embeddings
            synergy_bias = synergy_bias.view(B * self.nhead, seq_len, seq_len)
        else:
            synergy_bias = None

        # 3. Encode
        # Note: We pass synergy_bias as `mask` only if enabled.
        encoded = self.transformer_encoder(emb, mask=synergy_bias, src_key_padding_mask=padding_mask)

        # 4. Pooling
        # Use the CLS token (Index 0) representation
        cls_token = encoded[:, 0, :]

        # 5. Heads
        policy_logits = self.policy_head(cls_token)
        value = torch.tanh(self.value_head(cls_token))

        return policy_logits, value
