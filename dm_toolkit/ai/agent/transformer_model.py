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
        def tensor(*args, **kwargs):
            class _T:
                def expand(self, *args): return self
            return _T()
        class no_grad:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        @staticmethod
        def arange(*args): return torch.Tensor()
        @staticmethod
        def cat(*args, **kwargs): return torch.Tensor()
        @staticmethod
        def stack(*args, **kwargs): return torch.Tensor()

    class nn:
        class Module:
            def __init__(self): pass
            def parameters(self): return []
            def __call__(self, *args, **kwargs): return args[0]
        class Embedding(Module):
            def __init__(self, *args, **kwargs):
                self.weight = torch.Tensor()
                self.num_embeddings = args[0]
                self.embedding_dim = args[1]
        class Parameter:
            def __init__(self, *args, **kwargs): pass
        class TransformerEncoderLayer(Module):
            def __init__(self, *args, **kwargs): pass
        class TransformerEncoder(Module):
            def __init__(self, *args, **kwargs): pass
        class Sequential(Module):
            def __init__(self, *args, **kwargs):
                self.layers = args
            def __getitem__(self, idx): return self.layers[idx]
            def __setitem__(self, idx, val): pass
        class LayerNorm(Module):
            def __init__(self, *args, **kwargs): pass
        class Linear(Module):
            def __init__(self, *args, **kwargs):
                self.in_features = args[0]
                self.out_features = args[1]
                self.weight = torch.Tensor()
                self.bias = torch.Tensor()
        class GELU(Module):
            pass
        class init:
            @staticmethod
            def xavier_uniform_(p): pass
            @staticmethod
            def normal_(p, std=1.0): pass

    class F:
        @staticmethod
        def cross_entropy(*args, **kwargs): return 0.0
        @staticmethod
        def mse_loss(*args, **kwargs): return 0.0

import math
from typing import Optional, Tuple, List, Dict, Any
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


class DuelTransformerWithActionEmbedding(nn.Module):
    """
    Hierarchical Action Space Transformer.
    Separates action prediction into 'Type' and 'Parameters'.
    """
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

        # State Encoder (Shared with standard Transformer)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action Type Embedding
        self.action_type_embedding = nn.Embedding(num_action_types, d_model // 2)

        # Action Type Head (Classifier)
        self.action_type_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_action_types)
        )

        # Parameter Prediction Head
        self.action_param_head = nn.Sequential(
            nn.LayerNorm(d_model + d_model // 2),  # State + Type Embedding
            nn.Linear(d_model + d_model // 2, max_params_per_action)
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

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
        action_type_logits = self.action_type_head(state_repr)  # [B, num_action_types]

        # Step 2: Parameter Prediction for each type
        param_logits = []
        for action_type_id in range(self.action_type_embedding.num_embeddings):
            type_emb = self.action_type_embedding(
                torch.tensor([action_type_id], device=x.device).expand(B)
            )  # [B, d_model//2]

            combined = torch.cat([state_repr, type_emb], dim=-1)  # [B, d_model + d_model//2]
            param_logit = self.action_param_head(combined)  # [B, max_params]
            param_logits.append(param_logit)

        param_logits = torch.stack(param_logits, dim=1)  # [B, num_action_types, max_params]

        value = torch.tanh(self.value_head(state_repr))

        return action_type_logits, param_logits, value


def compute_loss_hierarchical(model_output, target_actions, target_values):
    """
    Computes loss for the hierarchical model.
    model_output: (action_type_logits, param_logits, value_pred)
    target_actions: [B, 2] (type_id, param_index)
    target_values: [B]
    """
    action_type_logits, param_logits, value_pred = model_output

    # target_actions: [B, 2] -> type_id, param_index
    target_type = target_actions[:, 0].long()
    target_param = target_actions[:, 1].long()

    # Type Classification Loss
    type_loss = F.cross_entropy(action_type_logits, target_type)

    # Parameter Prediction Loss (only for the correct type)
    B = action_type_logits.size(0)
    # param_logits: [B, num_types, max_params]
    selected_param_logits = param_logits[torch.arange(B), target_type]  # [B, max_params]
    param_loss = F.cross_entropy(selected_param_logits, target_param)

    # Value Loss
    value_loss = F.mse_loss(value_pred.squeeze(), target_values)

    total_loss = type_loss + param_loss + value_loss
    return total_loss


def extend_action_types(model: DuelTransformerWithActionEmbedding, num_new_types: int):
    """
    Extends the action type embedding and classification head.
    """
    old_embedding = model.action_type_embedding
    new_embedding = nn.Embedding(
        old_embedding.num_embeddings + num_new_types,
        old_embedding.embedding_dim
    )

    # Copy existing weights
    with torch.no_grad():
        new_embedding.weight[:old_embedding.num_embeddings] = old_embedding.weight
        # Initialize new types
        nn.init.normal_(new_embedding.weight[old_embedding.num_embeddings:], std=0.02)

    model.action_type_embedding = new_embedding

    # Extend action_type_head
    old_head = model.action_type_head[-1]
    new_head = nn.Linear(old_head.in_features, old_head.out_features + num_new_types)

    with torch.no_grad():
        new_head.weight[:old_head.out_features] = old_head.weight
        new_head.bias[:old_head.out_features] = old_head.bias

    model.action_type_head[-1] = new_head


def encode_action_hierarchical(action_dict: Dict[str, Any]) -> List[int]:
    """
    Encodes an action dictionary into [type_id, param_index].
    """
    action_type_map = {
        'PASS': 0,
        'MANA_CHARGE': 1,
        'PLAY_FROM_ZONE': 2,
        'PLAY_CARD': 2, # Alias
        'ATTACK_PLAYER': 3,
        'ATTACK_CREATURE': 4,
        # Extend as needed
    }

    a_type = action_dict.get('type', 'PASS')
    if isinstance(a_type, int):
        # If type is already int (legacy enum?), try to map common ones or assume direct mapping if valid
        # This is a fallback/hack. Better to ensure string or standard enum.
        # Here we assume string for safety with the map.
        # But if it's an int, we might need reverse lookup or standard map.
        pass

    type_id = action_type_map.get(a_type, 0)
    param_idx = action_dict.get('slot_index', 0)
    if param_idx is None:
        param_idx = 0

    return [type_id, param_idx]
