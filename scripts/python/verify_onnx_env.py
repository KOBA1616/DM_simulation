import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime
import os
import sys

# Add bin to path to import dm_ai_module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

# Try to import dm_toolkit to get access to dm_ai_module if it's wrapped there, 
# or import directly if it's a pyd in bin/ or build/
# The build output said: Linking CXX shared module dm_ai_module.cp313-win_amd64.pyd
# It is likely in build/ directory.
build_path = os.path.join(project_root, 'build')
if build_path not in sys.path:
    sys.path.append(build_path)

# Add project root to path to import dm_toolkit
if project_root not in sys.path:
    sys.path.append(project_root)

# Import dm_toolkit to setup DLL paths
try:
    import dm_toolkit
    print("Successfully imported dm_toolkit")
except ImportError as e:
    print(f"Failed to import dm_toolkit: {e}")

try:
    import dm_ai_module
    print("Successfully imported dm_ai_module")
    print(f"Module file: {dm_ai_module.__file__}")
    print(f"Module contents: {dir(dm_ai_module)}")
except ImportError as e:
    print(f"Failed to import dm_ai_module: {e}")
    # Try to find where it is
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Input size matches TensorConverter::INPUT_SIZE (need to check what it is, but for now assume something or make it dynamic)
        # Actually, OnnxModel in C++ just takes whatever input size the model expects usually, 
        # but NeuralEvaluator::evaluate converts states to TensorConverter::INPUT_SIZE.
        # Let's check TensorConverter::INPUT_SIZE. 
        # For verification of loading, the input size might not matter until we call evaluate.
        # But let's try to be correct.
        self.fc = nn.Linear(10, 2) 

    def forward(self, x):
        return self.fc(x)

def create_dummy_onnx(path):
    model = SimpleModel()
    model.eval()
    dummy_input = torch.randn(1, 10)
    
    # Export
    torch.onnx.export(model, dummy_input, path, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Exported dummy ONNX model to {path}")

def verify_cpp_loading(model_path):
    print("Verifying C++ loading...")
    # We need a card_db to initialize NeuralEvaluator
    # dm_ai_module.NeuralEvaluator(card_db)
    
    # We can try to load the JSON loader to get a card_db, or just pass an empty map if bindings allow
    # The binding expects: const std::map<CardID, CardDefinition>&
    # In python: dict
    
    card_db = {} 
    
    try:
        evaluator = dm_ai_module.NeuralEvaluator(card_db)
        print("Created NeuralEvaluator")
        
        evaluator.load_model(model_path)
        print("Successfully called load_model")
        
    except Exception as e:
        print(f"Error during C++ verification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    model_path = "dummy_model.onnx"
    create_dummy_onnx(model_path)
    
    # Verify with python onnxruntime first
    try:
        ort_session = onnxruntime.InferenceSession(model_path)
        print("Verified Python onnxruntime loading")
    except Exception as e:
        print(f"Python onnxruntime failed: {e}")
        sys.exit(1)
        
    verify_cpp_loading(model_path)
    
    print("ONNX Environment Verification Complete: SUCCESS")
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
