#!/bin/bash
set -e

# Duel Masters AI - Phase 4: Transformer Architecture Workflow
# This script demonstrates the full pipeline for training and using the Transformer model.

echo "--- Step 1: Building C++ Module ---"
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cp dm_ai_module.cpython-*.so ../bin/ 2>/dev/null || mkdir -p ../bin && cp dm_ai_module*.so ../bin/ 2>/dev/null
cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)/bin

echo "--- Step 2: Collecting Training Data (Tokens) ---"
# Collects episodes using the Heuristic Agent, saving sequences of tokens.
python3 python/training/collect_training_data.py --episodes 100 --mode transformer --output transformer_data.npz

echo "--- Step 3: Training Transformer Model ---"
# Trains the NetworkV2 (Linear Attention Transformer) on the collected data.
python3 dm_toolkit/training/train_simple.py --data_files transformer_data.npz --save model_transformer.pth --network_type transformer --epochs 5

echo "--- Step 4: Verifying Performance ---"
# Runs the trained model in a self-play scenario to verify inference flow.
python3 python/training/verify_performance.py --model model_transformer.pth --model_type transformer --sims 10 --episodes 5

echo "--- Phase 4 Complete ---"
echo "Model saved to: model_transformer.pth"
