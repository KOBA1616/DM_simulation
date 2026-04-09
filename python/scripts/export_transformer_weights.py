
import torch
import numpy as np
import argparse
import os
import sys

# Add bin/python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')
sys.path.append(bin_path)
sys.path.append(python_path)

from dm_toolkit.ai.agent.transformer_network import NetworkV2

def export_weights(model_path, output_path):
    print(f"Loading model from {model_path}...")
    # Initialize dummy model to load state dict (params don't matter much as we overwrite)
    # We need to guess or know the params. NetworkV2 default params:
    # embedding_dim=256, depth=6, heads=8, vocab=1000...
    # Better to load state dict and check shapes if possible, or use the training script logic to init.
    # For now, let's assume standard params or try to infer from checkpoint if stored.

    device = torch.device('cpu')
    state_dict = torch.load(model_path, map_location=device)

    # Infer params from state dict shapes
    vocab_size, dim = state_dict['card_embedding.weight'].shape
    max_seq_len = state_dict['pos_embedding'].shape[1]

    # Depth? Count layers.0. ...
    depth = 0
    while f'layers.{depth}.norm1.weight' in state_dict:
        depth += 1

    action_space = state_dict['policy_head.weight'].shape[0]

    print(f"Detected params: Dim={dim}, Depth={depth}, Vocab={vocab_size}, MaxSeq={max_seq_len}, Actions={action_space}")

    model = NetworkV2(embedding_dim=dim, depth=depth, heads=8, input_vocab_size=vocab_size, max_seq_len=max_seq_len, action_space=action_space)
    model.load_state_dict(state_dict)
    model.eval()

    weights_list = []

    def add_tensor(t):
        weights_list.append(t.detach().numpy().flatten())

    def add_linear(weight, bias):
        # Transpose weight from [Out, In] to [In, Out]
        add_tensor(weight.t())
        if bias is not None:
            add_tensor(bias)

    # Order matching C++ load_weights
    add_tensor(model.card_embedding.weight)
    add_tensor(model.pos_embedding) # [1, Seq, Dim] -> Flatten

    for layer in model.layers:
        # Norm1
        add_tensor(layer.norm1.weight)
        add_tensor(layer.norm1.bias)
        # Attn QKV
        # PyTorch Linear weight is [Out, In]
        # Our C++ expects [In, Out]
        add_linear(layer.attn.to_qkv.weight, None) # Bias is False

        # Attn Out
        add_linear(layer.attn.to_out[0].weight, layer.attn.to_out[0].bias)

        # Norm2
        add_tensor(layer.norm2.weight)
        add_tensor(layer.norm2.bias)

        # FFN1
        add_linear(layer.ffn[0].weight, layer.ffn[0].bias)

        # FFN2
        add_linear(layer.ffn[2].weight, layer.ffn[2].bias)

    # Norm
    add_tensor(model.norm.weight)
    add_tensor(model.norm.bias)

    # Heads
    add_linear(model.policy_head.weight, model.policy_head.bias)

    add_linear(model.value_head[0].weight, model.value_head[0].bias)
    add_linear(model.value_head[2].weight, model.value_head[2].bias)

    # Save to binary
    all_data = np.concatenate(weights_list).astype(np.float32)
    print(f"Total weights: {all_data.size} floats")

    all_data.tofile(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .pth model")
    parser.add_argument("output", help="Path to output .bin file")
    args = parser.parse_args()

    export_weights(args.model, args.output)
