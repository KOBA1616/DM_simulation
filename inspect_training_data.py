#!/usr/bin/env python3
"""
Inspect current training data format to determine if Transformer-compatible tokens are available.
"""
import os
import sys
import glob
import numpy as np

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def inspect_npz_files():
    """Search for and inspect all .npz training data files."""
    print("=" * 80)
    print("Training Data Format Inspector")
    print("=" * 80)
    
    # Search for npz files
    search_patterns = [
        "data/training*.npz",
        "data/**/training*.npz",
        "archive/data/training*.npz",
        "archive/**/training*.npz"
    ]
    
    found_files = []
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        found_files.extend(matches)
    
    found_files = list(set(found_files))  # Remove duplicates
    
    if not found_files:
        print("\n❌ No .npz training data files found.")
        print("\nSearched patterns:")
        for pattern in search_patterns:
            print(f"  - {pattern}")
        return False
    
    print(f"\n✅ Found {len(found_files)} training data file(s):\n")
    
    for file_path in sorted(found_files):
        print(f"\n{'='*80}")
        print(f"File: {file_path}")
        print(f"Size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
        print(f"{'='*80}")
        
        try:
            data = np.load(file_path, allow_pickle=True)
            print(f"\nKeys in file: {list(data.files)}\n")
            
            for key in data.files:
                arr = data[key]
                print(f"  {key:20s} | shape={str(arr.shape):30s} | dtype={str(arr.dtype):15s}")
                
                # Show sample values for small arrays
                if arr.size <= 20:
                    print(f"                         | values: {arr.flatten()[:10]}")
                elif arr.ndim == 1 and arr.size <= 1000:
                    print(f"                         | sample: {arr[:5]}...")
                elif arr.ndim >= 2:
                    print(f"                         | first row: {arr[0, :min(5, arr.shape[1])]}")
            
            print()
            
            # Specific checks for Transformer compatibility
            has_tokens = 'tokens' in data.files
            has_states = 'states' in data.files
            has_policies = 'policies' in data.files or 'policy' in data.files
            has_values = 'values' in data.files or 'value' in data.files
            
            print("Transformer Compatibility Check:")
            print(f"  ✅ Token sequences available: {has_tokens}")
            print(f"  ✅ State vectors available: {has_states}")
            print(f"  ✅ Policy targets available: {has_policies}")
            print(f"  ✅ Value targets available: {has_values}")
            
            if has_tokens:
                tokens = data['tokens']
                print(f"\n  Token Shape Details:")
                print(f"    - Number of samples: {tokens.shape[0]}")
                if tokens.ndim >= 2:
                    print(f"    - Sequence length (fixed): {tokens.shape[1]}")
                    print(f"    - Max token value: {np.max(tokens)}")
                    print(f"    - Min token value: {np.min(tokens)}")
                    print(f"    - Number of PAD tokens (0): {np.sum(tokens == 0)}")
            
            if has_states:
                states = data['states']
                print(f"\n  State Vector Details:")
                print(f"    - Number of samples: {states.shape[0]}")
                print(f"    - Feature dimension: {states.shape[1] if states.ndim >= 2 else 'N/A'}")
            
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
    
    return True

def check_training_source():
    """Check how training data should be generated."""
    print(f"\n{'='*80}")
    print("Training Data Generation Options")
    print(f"{'='*80}\n")
    
    print("Based on current implementation, training data can be generated via:\n")
    
    print("1. collect_training_data.py")
    if os.path.exists("dm_toolkit/training/collect_training_data.py"):
        print("   ✅ Found in dm_toolkit/training/")
        with open("dm_toolkit/training/collect_training_data.py", "r") as f:
            content = f.read()
            if "convert_to_sequence" in content:
                print("   ✅ Already supports token sequence generation")
            elif "TensorConverter" in content or "dm_ai_module" in content:
                print("   ⚠️  Uses C++ TensorConverter - may need token conversion")
    else:
        print("   ❌ Not found")
    
    print("\n2. Self-play games via scenario_runner.py or self_play.py")
    for script in ["dm_toolkit/training/scenario_runner.py", "dm_toolkit/training/self_play.py"]:
        if os.path.exists(script):
            print(f"   ✅ {script} exists")
    
    print("\n3. Existing archived data in archive/data/")
    if os.path.isdir("archive/data"):
        files = os.listdir("archive/data")
        print(f"   ✅ Found {len(files)} files")

if __name__ == "__main__":
    success = inspect_npz_files()
    check_training_source()
    
    if not success:
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)
        print("""
If no training data found:
1. Run: python dm_toolkit/training/collect_training_data.py
   This will generate training samples from self-play games.

2. Or use scenario-based data:
   python dm_toolkit/training/scenario_runner.py --scenarios all --output data/training_data.npz

3. Expected output format:
   - 'tokens': [num_samples, seq_len] of integer token IDs
   - 'states': [num_samples, feature_dim] (optional, for MLP comparison)
   - 'policies': [num_samples, action_dim] policy targets
   - 'values': [num_samples, 1] value targets
        """)
