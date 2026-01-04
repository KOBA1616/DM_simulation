import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, cast

class AlphaZeroMLP(nn.Module):
    def __init__(self, input_size: int, action_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        
        # Policy Head
        self.policy_head = nn.Linear(1024, action_size)
        
        # Value Head
        self.value_head = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        policy = self.policy_head(x) # Logits
        value = torch.tanh(self.value_head(x)) # [-1, 1]
        
        return policy, value

# Alias for backward compatibility
AlphaZeroNetwork = AlphaZeroMLP

class LinearAttention(nn.Module):
    """
    Linear Attention implementation for Phase 4 Architecture Update.
    Reduces complexity from O(N^2) to O(N) for variable length sequence processing.

    References:
        - "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (Katharopoulos et al.)
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, SeqLen, Dim)
            mask: Optional boolean mask of shape (Batch, SeqLen) where True indicates valid tokens.
        """
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # Split heads
        # q, k, v shape: (b, n, h*d) -> (b, n, h, d) -> (b, h, n, d)
        q, k, v = map(lambda t: t.view(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))

        # Linear Attention Logic (elu + 1) ensures positive feature map
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Apply mask if provided
        if mask is not None:
            # mask shape: (b, n)
            # k shape: (b, h, n, d)
            # We want to mask along dimension n (dim 2 in k).
            # Expand mask to (b, 1, n, 1) to broadcast against (b, h, n, d)
            mask = mask[:, None, :, None]
            k = k.masked_fill(~mask, 0.)
            v = v.masked_fill(~mask, 0.)

        # KV calculation (O(N) step)
        # k: (b, h, n, d)
        # v: (b, h, n, e) -> here e=d
        # kv: (b, h, d, e)
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        # Output calculation
        # q: (b, h, n, d)
        # out: (b, h, n, e)
        out = torch.einsum('b h n d, b h d e -> b h n e', q, kv)

        # Denominator for normalization
        # k.sum(dim=2): (b, h, d) -> sum over sequence length
        # q: (b, h, n, d)
        # z: (b, h, n)
        z = 1 / (torch.einsum('b h n d, b h d -> b h n', q, k.sum(dim=2)) + 1e-6)
        out = out * z.unsqueeze(-1)

        # Merge heads
        # out: (b, h, n, d) -> (b, n, h, d) -> (b, n, h*d)
        # out.shape[-1] is d. We need h*d.
        out = out.transpose(1, 2).reshape(out.shape[0], -1, h * out.shape[-1])
        return cast(torch.Tensor, self.to_out(out))

class TransformerBlock(nn.Module):
    """
    Standard Transformer Block adapted with Linear Attention.
    Structure: Pre-Norm -> Attention -> Add -> Pre-Norm -> FFN -> Add
    """
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class AlphaZeroTransformer(nn.Module):
    """
    Transformer-based Dual Network (Policy & Value) for Duel Masters AI.
    Replaces the fixed-size ResNet with a sequence-based architecture using Linear Attention.
    """
    def __init__(self, action_size: int, embedding_dim: int = 256, depth: int = 6, heads: int = 8,
                 vocab_size: int = 10000, max_seq_len: int = 200):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Embeddings for Card IDs, Zones, etc.
        # vocab_size + 1000 safety buffer as per verification script convention, though strict size is better
        self.card_embedding = nn.Embedding(vocab_size + 1000, embedding_dim)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim))

        self.layers = nn.ModuleList([
            TransformerBlock(embedding_dim, heads=heads, dim_head=embedding_dim//heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        # Heads
        self.policy_head = nn.Linear(embedding_dim, action_size)
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # No Tanh here yet, applied in forward
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x: Input token indices [Batch, SeqLen]
            mask: Optional validity mask [Batch, SeqLen]

        Returns:
            policy_logits: [Batch, ActionSpace] (Logits)
            value: [Batch, 1] (Tanh activation)
        """
        b, n = x.shape

        # Robustly handle sequences exceeding max_seq_len
        if n > self.max_seq_len:
            x = x[:, :self.max_seq_len]
            if mask is not None:
                mask = mask[:, :self.max_seq_len]
            n = self.max_seq_len

        x = self.card_embedding(x)

        # Add positional embedding, slicing to current sequence length
        # At this point n <= self.max_seq_len is guaranteed
        x += self.pos_embedding[:, :n]

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Global pooling (Mean pooling with mask awareness)
        if mask is not None:
             # Valid length averaging
             x_sum = (x * mask.unsqueeze(-1)).sum(dim=1)
             x_len = mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
             pooled = x_sum / x_len
        else:
             pooled = x.mean(dim=1)

        policy_logits = self.policy_head(pooled)
        value = torch.tanh(self.value_head(pooled))

        return policy_logits, value
