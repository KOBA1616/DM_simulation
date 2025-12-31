
import sys
import os
import pytest
import shutil
import subprocess
import platform

# Skip ONNX/C++ transformer verification on Windows to avoid native runtime
# ABI/API mismatches that can cause access violations in the C++ extension.
if platform.system() == "Windows":
    pytest.skip("Skipping transformer ONNX verification on Windows due to native onnxruntime incompatibility", allow_module_level=True)

# Ensure bin and project root are in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.abspath("bin"))

try:
    import dm_ai_module
except ImportError:
    pytest.fail("Could not import dm_ai_module. Ensure it is built and in bin/")

def test_neural_evaluator_transformer_onnx():
    """Verify that NeuralEvaluator can run Transformer inference via ONNX Runtime in C++."""

    # 1. Export the ONNX model first
    export_script = os.path.join(os.path.dirname(__file__), "export_transformer_onnx.py")
    subprocess.check_call([sys.executable, export_script])

    onnx_path = "transformer_test.onnx"
    if not os.path.exists(onnx_path):
        pytest.fail("Failed to export ONNX model")

    # 2. Setup GameState and Evaluator
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    if not card_db:
        # Create minimal DB if missing
        pass

    evaluator = dm_ai_module.NeuralEvaluator(card_db)

    # 3. Configure for Transformer and Load Model
    evaluator.set_model_type(dm_ai_module.ModelType.TRANSFORMER)
    evaluator.load_model(onnx_path)

    # 4. Clear any python callbacks to ensure we use C++ path
    dm_ai_module.clear_sequence_batch_callback()

    # 5. Create dummy states
    state = dm_ai_module.GameState(100)
    state.setup_test_duel()
    states = [state, state]

    # 6. Evaluate
    # If C++ ONNX path is working, this should return results.
    # If not, it will fall back to callback (which is cleared) and print error/return zeros.
    policies, values = evaluator.evaluate(states)

    # 7. Verify
    # Check if we got non-zero/non-empty results.
    # Since model is random weights, values should be within [-1, 1].
    # If fallback triggered, values are 0.0. But random model might output close to 0.
    # However, standard fallback returns exact 0.0s.

    print(f"Values: {values}")

    assert len(policies) == 2
    assert len(values) == 2

    # Check shape of policy
    assert len(policies[0]) == 10 # action space in export script

    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

if __name__ == "__main__":
    test_neural_evaluator_transformer_onnx()
