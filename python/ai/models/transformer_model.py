import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DuelTransformer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 vocab_size=4096,
                 max_seq_len=2048,
                 action_space_size=600):
        super(DuelTransformer, self).__init__()

        self.d_model = d_model

        # 1. Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Increase PE max_len to handle Meta token insertion
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 1)

        # 2. Meta Embedding (Global Context)
        self.meta_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # 3. Synergy Bias Matrix
        self.synergy_matrix = nn.Parameter(torch.randn(vocab_size, vocab_size) * 0.01)

        # 4. Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 5. Heads
        self.policy_head = nn.Linear(d_model, action_space_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)

        batch_size = src.size(0)
        meta_token = self.meta_embedding.expand(batch_size, -1, -1)
        x = torch.cat([meta_token, x], dim=1) # Length N+1

        # Positional Encoding will handle N+1 if max_len is sufficient
        # If input src was already max_seq_len, x is max_seq_len+1
        x = self.pos_encoder(x)

        # Synergy Bias Mask
        # src_extended must align with x
        meta_indices = torch.zeros((batch_size, 1), dtype=src.dtype, device=src.device)
        src_extended = torch.cat([meta_indices, src], dim=1)

        # If x was truncated by PE (rare but possible if src was huge), align src_extended
        if x.size(1) < src_extended.size(1):
            src_extended = src_extended[:, :x.size(1)]

        src_extended = torch.clamp(src_extended, 0, self.embedding.num_embeddings - 1)

        synergy_bias = self.synergy_matrix[src_extended.unsqueeze(2), src_extended.unsqueeze(1)]

        nhead = self.transformer_encoder.layers[0].self_attn.num_heads
        synergy_bias = synergy_bias.repeat_interleave(nhead, dim=0)

        output = self.transformer_encoder(x, mask=synergy_bias, src_key_padding_mask=None)

        cls_output = output[:, 0, :]

        policy_logits = self.policy_head(cls_output)
        value = torch.tanh(self.value_head(cls_output))

        return policy_logits, value

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
             x = x[:, :self.pe.size(1), :]
             seq_len = self.pe.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
