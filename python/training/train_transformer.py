import os
import sys
import argparse
import glob

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from dm_toolkit.training.train_simple import train_pipeline
except ImportError as e:
    print(f"Error importing dm_toolkit: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer model for Duel Masters AI")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing .npz training data")
    parser.add_argument("--data_files", nargs='+', help="Specific .npz files to train on")
    parser.add_argument("--model", type=str, default=None, help="Path to existing model to resume training")
    parser.add_argument("--output", type=str, default="transformer_model.pth", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    args = parser.parse_args()

    files = []
    if args.data_files:
        files.extend(args.data_files)

    if args.data_dir:
        if os.path.exists(args.data_dir):
            found = glob.glob(os.path.join(args.data_dir, "*.npz"))
            files.extend(found)
        else:
             print(f"Warning: Data directory {args.data_dir} does not exist.")

    # Remove duplicates
    files = list(set(files))

    if not files:
        print("No training data found. Use --data_dir or --data_files.")
        sys.exit(1)

    print(f"Starting Transformer training with {len(files)} files...")
    print(f"Output model: {args.output}")

    # Force network_type='transformer' to ensure we use the Transformer path in Trainer
    train_pipeline(files, args.model, args.output, epochs=args.epochs, network_type='transformer')
