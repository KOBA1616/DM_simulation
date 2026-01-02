import sys
import os
import torch
import torch.nn as nn

# Ensure bin and project root are in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from dm_toolkit.ai.agent.transformer_network import NetworkV2
except ImportError:
    print("Could not import NetworkV2")
    sys.exit(1)

def export_onnx():
    vocab_size = 2000
    action_space = 10
    batch_size = 2
    seq_len = 20
    model = NetworkV2(input_vocab_size=vocab_size, action_space=action_space)
    model.eval()

    # Create dummy input tokens [B, SeqLen]
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Create dummy mask [B, SeqLen]
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    # Export
    output_path = "transformer_test.onnx"

    # Dynamic axes for variable sequence length
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'seq_len'},
        'mask': {0: 'batch_size', 1: 'seq_len'},
        'policy': {0: 'batch_size'},
        'value': {0: 'batch_size'}
    }

    try:
        torch.onnx.export(
            model,
            (x, mask),
            output_path,
            input_names=['input_ids', 'mask'],
            output_names=['policy', 'value'],
            dynamic_axes=dynamic_axes,
            opset_version=14
        )
        print(f"Successfully exported to {output_path}")

        # Verify with onnx (if installed)
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        print("\nInput definitions:")
        for inp in onnx_model.graph.input:
            print(f"Name: {inp.name}, Type: {inp.type}")

    except Exception as e:
        print(f"Export failed: {e}")

if __name__ == '__main__':
    export_onnx()
