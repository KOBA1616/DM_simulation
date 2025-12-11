import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    """
    Linear Attention implementation for Phase 4 Architecture Update.
    Reduces complexity from O(N^2) to O(N) for variable length sequence processing.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # Split heads
        q, k, v = map(lambda t: t.view(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))

        # Linear Attention Logic (elu + 1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Apply mask if provided (not fully implemented for variable length yet)
        if mask is not None:
            mask = mask[:, None, None, :]
            k = k.masked_fill(~mask, 0.)
            v = v.masked_fill(~mask, 0.)

        # KV calculation
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        # Output calculation
        out = torch.einsum('b h n d, b h d e -> b h n e', q, kv)

        # Denominator for normalization
        z = 1 / (torch.einsum('b h n d, b h d -> b h n', q, k.sum(dim=2)) + 1e-6)
        out = out * z.unsqueeze(-1)

        # Merge heads
        out = out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[-1])
        return self.to_out(out)

class NetworkV2(nn.Module):
    """
    Transformer-based Dual Network (Policy & Value) for Duel Masters AI.
    Replaces the fixed-size ResNet with a sequence-based architecture.
    """
    def __init__(self, embedding_dim=256, depth=6, heads=8, input_vocab_size=1000, max_seq_len=200, action_space=600):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Embeddings for Card IDs, Zones, etc.
        self.card_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim))

        self.layers = nn.ModuleList([
            LinearAttention(embedding_dim, heads=heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        # Heads
        self.policy_head = nn.Linear(embedding_dim, action_space)
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x, mask=None):
        # x: [Batch, SeqLen] (indices of cards/features)

        b, n = x.shape
        x = self.card_embedding(x)
        x += self.pos_embedding[:, :n]

        for layer in self.layers:
            x = x + layer(x, mask)

        x = self.norm(x)

        # Global pooling (e.g., CLS token or mean)
        # For now, simple mean pooling over the sequence
        if mask is not None:
             # Valid length averaging
             x_sum = (x * mask.unsqueeze(-1)).sum(dim=1)
             x_len = mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
             pooled = x_sum / x_len
        else:
             pooled = x.mean(dim=1)

        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)

        return F.softmax(policy_logits, dim=-1), value
