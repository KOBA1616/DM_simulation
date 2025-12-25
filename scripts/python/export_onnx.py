
import os
import sys
import torch
import torch.onnx
import argparse

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)

# Add project root to path for dm_toolkit import
if project_root not in sys.path:
    sys.path.append(project_root)

from dm_toolkit.ai.agent.network import AlphaZeroNetwork

# Default constants based on the system
DEFAULT_INPUT_SIZE = 205  # Approximate, will verify
DEFAULT_ACTION_SIZE = 600 # Defined in ActionEncoder

def export_to_onnx(model_path, output_path, input_size=DEFAULT_INPUT_SIZE, action_size=DEFAULT_ACTION_SIZE):
    """
    Exports a PyTorch model to ONNX format.
    """
    print(f"Exporting model from {model_path} to {output_path}...")
    print(f"Input Size: {input_size}, Action Size: {action_size}")

    device = torch.device("cpu") # ONNX export usually done on CPU

    # Initialize model
    model = AlphaZeroNetwork(input_size, action_size).to(device)

    # Load weights if available
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded model weights.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using random weights instead.")
    else:
        print(f"Model file {model_path} not found. Using random weights.")

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, input_size, device=device)

    # Export
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=18,  # Increased to match modern torch default
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy', 'value'],
            dynamic_axes={'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
        )
        print(f"Successfully exported to {output_path}")
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model", type=str, default="data/model_latest.pth", help="Path to input .pth model")
    parser.add_argument("--output", type=str, default="data/model.onnx", help="Path to output .onnx file")
    parser.add_argument("--input_size", type=int, default=DEFAULT_INPUT_SIZE, help="Input tensor size")
    parser.add_argument("--action_size", type=int, default=DEFAULT_ACTION_SIZE, help="Action tensor size")

    args = parser.parse_args()

    export_to_onnx(args.model, args.output, args.input_size, args.action_size)
