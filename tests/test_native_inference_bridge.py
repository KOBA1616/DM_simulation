
import sys
import os

# Ensure path is set to import local modules
sys.path.append(os.getcwd())

from dm_toolkit.engine.compat import EngineCompat
import dm_ai_module

def fail(msg):
    print(f"FAILED: {msg}")
    sys.exit(1)

def test_missing_inference_components():
    """
    Verifies that the components required for simulation/inference exist in dm_ai_module
    (or its fallback).
    """
    print("Testing dm_ai_module components...")

    # 1. Check ScenarioConfig
    if not hasattr(dm_ai_module, 'ScenarioConfig'):
        fail("dm_ai_module.ScenarioConfig is missing")
    print("ScenarioConfig OK")

    # 2. Check TensorConverter
    if not hasattr(dm_ai_module, 'TensorConverter'):
        fail("dm_ai_module.TensorConverter is missing")

    state = dm_ai_module.GameState()
    # Check convert_to_tensor signature
    try:
        vec = dm_ai_module.TensorConverter.convert_to_tensor(state, 0, None)
        if not (isinstance(vec, list) or hasattr(vec, '__len__')):
            fail("TensorConverter.convert_to_tensor returned invalid type")
    except Exception as e:
        fail(f"TensorConverter.convert_to_tensor failed: {e}")
    print("TensorConverter OK")

    # 3. Check register_batch_inference_numpy
    if not hasattr(dm_ai_module, 'register_batch_inference_numpy'):
        fail("dm_ai_module.register_batch_inference_numpy is missing")

    # Test calling it
    try:
        dm_ai_module.register_batch_inference_numpy(None)
    except Exception as e:
        fail(f"register_batch_inference_numpy failed: {e}")
    print("register_batch_inference_numpy OK")

    # 4. Check ParallelRunner
    if not hasattr(dm_ai_module, 'ParallelRunner'):
        fail("dm_ai_module.ParallelRunner is missing")

    # Test instantiation
    try:
        runner = dm_ai_module.ParallelRunner(None, 10, 8)
    except Exception as e:
        fail(f"ParallelRunner instantiation failed: {e}")

    # Test play_games
    try:
        states = [dm_ai_module.GameState() for _ in range(2)]
        def eval_func(states):
            return [([1.0/600]*600, 0.0) for _ in states]

        results = runner.play_games(states, eval_func, 1.0, False, 1)
        if not isinstance(results, list):
            fail("play_games did not return a list")
        if len(results) != 2:
            fail(f"play_games returned {len(results)} results, expected 2")
    except Exception as e:
        fail(f"ParallelRunner.play_games failed: {e}")
    print("ParallelRunner OK")

    # 5. Check Evaluators
    if not hasattr(dm_ai_module, 'HeuristicEvaluator'):
        fail("dm_ai_module.HeuristicEvaluator is missing")
    if not hasattr(dm_ai_module, 'NeuralEvaluator'):
        fail("dm_ai_module.NeuralEvaluator is missing")
    print("Evaluators OK")

if __name__ == "__main__":
    test_missing_inference_components()
    print("ALL TESTS PASSED")
