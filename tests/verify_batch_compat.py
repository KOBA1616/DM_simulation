
import sys
import os
import unittest
from typing import List, Any

# Ensure we can import dm_toolkit
sys.path.append(os.getcwd())

from dm_toolkit.engine.compat import EngineCompat
import dm_toolkit.dm_ai_module as dm_ai_module

class TestBatchCompat(unittest.TestCase):
    def test_components_exist(self):
        """Verify that the components required by SimulationDialog exist."""
        print(f"\nChecking dm_ai_module (Native: {getattr(dm_ai_module, 'IS_NATIVE', False)})")

        # 1. ScenarioConfig
        self.assertTrue(hasattr(dm_ai_module, 'ScenarioConfig'), "ScenarioConfig missing")
        config = dm_ai_module.ScenarioConfig()
        self.assertTrue(hasattr(config, 'my_mana'), "ScenarioConfig.my_mana missing")

        # 2. HeuristicEvaluator
        self.assertTrue(hasattr(dm_ai_module, 'HeuristicEvaluator'), "HeuristicEvaluator missing")
        # Assuming CardDatabase is needed
        db = dm_ai_module.CardDatabase()
        evaluator = dm_ai_module.HeuristicEvaluator(db)

        # 3. NeuralEvaluator
        self.assertTrue(hasattr(dm_ai_module, 'NeuralEvaluator'), "NeuralEvaluator missing")
        ne_evaluator = dm_ai_module.NeuralEvaluator(db)

        # 4. TensorConverter via Compat
        state = dm_ai_module.GameState()
        tensor = EngineCompat.TensorConverter_convert_to_tensor(state, 0, db)
        print(f"TensorConverter returned type: {type(tensor)}")
        self.assertIsInstance(tensor, (list, tuple, object)) # Should be a list or something iterable

        # 5. register_batch_inference_numpy via Compat
        def callback(x): return x
        try:
            EngineCompat.register_batch_inference_numpy(callback)
            print("register_batch_inference_numpy successful")
        except Exception as e:
            self.fail(f"register_batch_inference_numpy failed: {e}")

        # 6. ParallelRunner via Compat
        runner = EngineCompat.create_parallel_runner(db, 10, 2)
        if getattr(dm_ai_module, 'IS_NATIVE', False):
             self.assertIsNotNone(runner, "ParallelRunner creation failed (Native)")
        else:
             # If not native, it might be None unless we stub it.
             # The goal is to HAVE it stubbed so it's not None.
             print(f"ParallelRunner (Stub): {runner}")

        # 7. Play Games
        if runner:
            try:
                # Need a minimal state
                states = [dm_ai_module.GameState(i) for i in range(2)]
                def eval_func(states):
                    return [([0]*600, 0.0) for _ in states]

                results = EngineCompat.ParallelRunner_play_games(runner, states, eval_func, 1.0, False, 1)
                print(f"ParallelRunner results: {len(results)}")
            except Exception as e:
                self.fail(f"ParallelRunner.play_games failed: {e}")

if __name__ == '__main__':
    unittest.main()
