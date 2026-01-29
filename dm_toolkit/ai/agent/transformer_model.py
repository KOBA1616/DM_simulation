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
        @staticmethod
        def ones(*args, **kwargs): return torch.Tensor()
        @staticmethod
        def zeros(*args, **kwargs): return torch.Tensor()
        @staticmethod
        def cat(*args, **kwargs): return torch.Tensor()
        class bool: pass
        @staticmethod
        def multinomial(*args, **kwargs): return torch.Tensor()
        @staticmethod
        def no_grad():
             class NoGrad:
                 def __enter__(self): pass
                 def __exit__(self, *args): pass
             return NoGrad()
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
        @staticmethod
        def softmax(*args, **kwargs): return torch.Tensor()

import math
from typing import Optional, Tuple, List
from dm_toolkit.ai.agent.synergy import SynergyGraph

class DuelTransformer(nn.Module):
    """
    Phase 8 Transformer Architecture (Sequence Understanding).

    Replaces the fixed-length feature vector with a token-based sequence model.
    Incorporates Synergy Bias Mask for card compatibility understanding.

    Updated with Dynamic Output Layer support (Idea 2).

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
    def __init__(self, vocab_size: int, action_dim: int, reserved_dim: int = 1024, d_model: int = 256, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 1024, max_len: int = 200, synergy_matrix_path: Optional[str] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.nhead = nhead
        self.action_dim = action_dim
        self.reserved_dim = reserved_dim

        if action_dim > reserved_dim:
            raise ValueError(f"action_dim {action_dim} cannot be larger than reserved_dim {reserved_dim}")

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
        # Phase Specific Policies
        self.main_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, reserved_dim)
        )

        # Active Action Mask (Buffer)
        # Tracks which dimensions are currently in use
        self.register_buffer(
            'active_action_mask',
            torch.cat([
                torch.ones(action_dim, dtype=torch.bool),
                torch.zeros(reserved_dim - action_dim, dtype=torch.bool)
            ])
        )
        self.mana_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim)
        )
        self.attack_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim)
        )

        # Alias for backward compatibility
        self.policy_head = self.main_head

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

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, phase_ids: Optional[torch.Tensor] = None, legal_action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, SeqLen] (Integer Token IDs)
            padding_mask: [Batch, SeqLen] (Boolean, True = Pad/Ignored) - Optional
            phase_ids: [Batch] (Integer Phase IDs) - Optional
            legal_action_mask: [Batch, reserved_dim] (Boolean, True = Legal) - Optional
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
        if phase_ids is not None:
            # Initialize with main head outputs
            policy_logits = self.main_head(cls_token)

            # Apply Mana Phase Head (Phase.MANA = 2)
            mask_mana = (phase_ids == 2)
            if mask_mana.any():
                # We only compute the head for the relevant subset to save compute?
                # Actually, slicing and computing might be slower if batch is small, but logically correct.
                # However, since we initialized policy_logits with main_head, we are overwriting.
                # A more efficient way if we had many heads:
                # policy_logits = torch.empty(B, self.policy_head[1].out_features, device=x.device, dtype=cls_token.dtype)
                # But initialization with main_head is fine for now.
                # Assign only to active dimension slice since phase heads match action_dim (or their specific size)
                out_dim = self.mana_head[-1].out_features
                policy_logits[mask_mana, :out_dim] = self.mana_head(cls_token[mask_mana])

            # Apply Attack Phase Head (Phase.ATTACK = 4)
            mask_attack = (phase_ids == 4)
            if mask_attack.any():
                out_dim = self.attack_head[-1].out_features
                policy_logits[mask_attack, :out_dim] = self.attack_head(cls_token[mask_attack])
        else:
            policy_logits = self.policy_head(cls_token)

        # ★ Masking Application
        # 1. Mask inactive dimensions
        # active_action_mask is [reserved_dim]
        # We expand it to [Batch, reserved_dim]
        inactive_mask = ~self.active_action_mask.unsqueeze(0).expand(B, -1)
        policy_logits = policy_logits.masked_fill(inactive_mask, -1e9)

        # 2. Mask illegal actions (Learning / Inference)
        if legal_action_mask is not None:
            illegal_mask = ~legal_action_mask
            policy_logits = policy_logits.masked_fill(illegal_mask, -1e9)

        value = torch.tanh(self.value_head(cls_token))

        return policy_logits, value

    def activate_reserved_actions(self, new_action_count: int) -> None:
        """Enables previously reserved action dimensions."""
        current_active = int(self.active_action_mask.sum().item())
        new_total = current_active + new_action_count

        if new_total > self.reserved_dim:
            raise ValueError(f"Exceeds reserved dimensions {self.reserved_dim}")

        self.active_action_mask[current_active:new_total] = True
        self.action_dim = new_total
        print(f"Action dimensions expanded: {current_active} -> {new_total}")

    def predict_action(self, state_tokens: torch.Tensor, legal_actions: Optional[List[int]] = None) -> Tuple[int, float]:
        """
        Helper for inference.
        Args:
            state_tokens: [1, SeqLen]
            legal_actions: List of legal action indices (optional)
        Returns:
            (action_idx, value)
        """
        self.eval()
        legal_mask = None
        if legal_actions is not None:
             legal_mask = torch.zeros(self.reserved_dim, dtype=torch.bool, device=state_tokens.device)
             # Filter indices within reserved range
             valid_indices = [idx for idx in legal_actions if 0 <= idx < self.reserved_dim]
             if valid_indices:
                 legal_mask[valid_indices] = True
             legal_mask = legal_mask.unsqueeze(0) # [1, reserved_dim]

        with torch.no_grad():
             policy_logits, value = self(state_tokens, legal_action_mask=legal_mask)

             # Softmax (safe with large negative numbers)
             probs = F.softmax(policy_logits, dim=-1)

             # Sample
             if probs.sum() == 0:
                 # Fallback if everything masked (shouldn't happen if legal_actions provided correctly)
                 action_idx = 0
             else:
                 action_idx = torch.multinomial(probs, 1).item()

        return action_idx, value.item()
