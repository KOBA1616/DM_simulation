import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DuelTransformer(nn.Module):
    """
    Phase 8 Transformer Architecture (BERT-like).

    Specs:
    - Architecture: Encoder-Only Transformer
    - d_model: 256
    - Layers: 6
    - Heads: 8
    - d_ff: 1024
    - Activation: GELU
    - Context Length: 512 (Max) - Currently adapted for flat input.
    """
    def __init__(self, input_dim, action_dim, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, seq_len=16):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # --- Legacy Feature Adapter ---
        # Projects flat input (approx 205 floats) into a synthetic sequence.
        # We project to (seq_len * d_model) to create a sequence of latent tokens.
        # This allows the Transformer to process the state as a set of features.
        self.input_projection = nn.Linear(input_dim, seq_len * d_model)

        # Positional Encoding (Learnable is often better for fixed-length latent sequences)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Heads ---
        # We use the mean of the sequence for the final prediction (Global Average Pooling)
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim)
        )

        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize parameters with specific logic if needed,
        # though PyTorch defaults are generally okay.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [Batch, InputDim] (Flat vector)

        batch_size = x.size(0)

        # 1. Adapt Flat Input to Sequence
        # [Batch, InputDim] -> [Batch, SeqLen * d_model]
        x = self.input_projection(x)

        # Reshape to [Batch, SeqLen, d_model]
        x = x.view(batch_size, self.seq_len, self.d_model)

        # 2. Add Positional Embeddings
        x = x + self.pos_embedding

        # 3. Transformer Encoder
        # Output: [Batch, SeqLen, d_model]
        x = self.transformer_encoder(x)

        # 4. Global Pooling (Mean)
        # [Batch, d_model]
        x_pooled = x.mean(dim=1)

        # 5. Heads
        policy_logits = self.policy_head(x_pooled)
        value = torch.tanh(self.value_head(x_pooled))

        return policy_logits, value
