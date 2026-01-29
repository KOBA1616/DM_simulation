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

class DuelTransformerWithActionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_action_types: int,
        max_params_per_action: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_len: int = 200
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # State Encoder
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action Type Embedding
        self.action_type_embedding = nn.Embedding(num_action_types, d_model // 2)

        # Action Classification Head
        self.action_type_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_action_types)
        )

        # Parameter Prediction Head
        self.action_param_head = nn.Sequential(
            nn.LayerNorm(d_model + d_model // 2),
            nn.Linear(d_model + d_model // 2, max_params_per_action)
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if 'pos_embedding' in name or 'action_type_embedding' in name:
                 if p.dim() > 1:
                     nn.init.normal_(p, std=0.02)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, padding_mask=None):
        B, S = x.shape

        # State Encoding
        emb = self.token_embedding(x)
        seq_len = min(S, self.max_len)
        pos = self.pos_embedding[:seq_len, :].unsqueeze(0).expand(B, seq_len, self.d_model)
        emb = emb[:, :seq_len, :] + pos
        encoded = self.transformer_encoder(emb, src_key_padding_mask=padding_mask)
        state_repr = encoded[:, 0, :]  # CLS token

        # Step 1: Action Type Selection
        action_type_logits = self.action_type_head(state_repr)

        # Step 2: Parameter Prediction
        param_logits = []
        for action_type_id in range(self.action_type_embedding.num_embeddings):
            type_emb = self.action_type_embedding(
                torch.tensor([action_type_id], device=x.device).expand(B)
            )
            combined = torch.cat([state_repr, type_emb], dim=-1)
            param_logit = self.action_param_head(combined)
            param_logits.append(param_logit)

        param_logits = torch.stack(param_logits, dim=1)  # [B, num_action_types, max_params]

        value = torch.tanh(self.value_head(state_repr))

        return action_type_logits, param_logits, value


def compute_loss_hierarchical(model_output, target_actions, target_values):
    """
    Computes loss for hierarchical action prediction.
    Args:
        model_output: (action_type_logits, param_logits, value_pred)
        target_actions: [B, 2] -> [action_type_id, param_index]
        target_values: [B]
    """
    action_type_logits, param_logits, value_pred = model_output

    target_type = target_actions[:, 0].long()
    target_param = target_actions[:, 1].long()

    # Type Loss
    type_loss = F.cross_entropy(action_type_logits, target_type)

    # Parameter Loss
    B = action_type_logits.size(0)
    selected_param_logits = param_logits[torch.arange(B), target_type]
    param_loss = F.cross_entropy(selected_param_logits, target_param)

    # Value Loss
    value_loss = F.mse_loss(value_pred.squeeze(), target_values)

    total_loss = type_loss + param_loss + value_loss
    return total_loss


def extend_action_types(model, num_new_types):
    """
    Extends the action type embedding and head to support new types.
    """
    old_embedding = model.action_type_embedding
    device = old_embedding.weight.device
    new_embedding = nn.Embedding(
        old_embedding.num_embeddings + num_new_types,
        old_embedding.embedding_dim
    ).to(device)

    with torch.no_grad():
        new_embedding.weight[:old_embedding.num_embeddings] = old_embedding.weight
        nn.init.normal_(new_embedding.weight[old_embedding.num_embeddings:], std=0.02)

    model.action_type_embedding = new_embedding

    old_head = model.action_type_head[-1]
    new_head = nn.Linear(old_head.in_features, old_head.out_features + num_new_types).to(device)
    with torch.no_grad():
        new_head.weight[:old_head.out_features] = old_head.weight
        new_head.bias[:old_head.out_features] = old_head.bias

    model.action_type_head[-1] = new_head


def encode_action_hierarchical(action_dict):
    """
    Encodes an action dictionary into [type_id, param_index].
    """
    action_type_map = {
        'PASS': 0,
        'MANA_CHARGE': 1,
        'PLAY_FROM_ZONE': 2,
    }

    # Handle dictionary or object
    t = action_dict.get('type') if isinstance(action_dict, dict) else getattr(action_dict, 'type', None)

    # Handle Enum or String
    if hasattr(t, 'name'):
        t_str = t.name
    else:
        t_str = str(t)

    # Normalize 'PLAY_CARD' to 'PLAY_FROM_ZONE' if needed, or just handle legacy
    if t_str == 'PLAY_CARD': # Legacy support
        t_str = 'PLAY_FROM_ZONE'

    type_id = action_type_map.get(t_str, 0)

    if isinstance(action_dict, dict):
        param_idx = int(action_dict.get('slot_index', 0))
    else:
        param_idx = int(getattr(action_dict, 'slot_index', 0))

    return [type_id, param_idx]
