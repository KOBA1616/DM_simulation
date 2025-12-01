import numpy as np
import dm_ai_module

def f(arr):
    print('DEBUG f called, arr.shape=', arr.shape, 'dtype=', arr.dtype)
    policies = np.zeros((arr.shape[0], dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE), dtype=np.float32)
    values = np.zeros((arr.shape[0],), dtype=np.float32)
    return policies, values

print('registering f')
dm_ai_module.register_batch_inference_numpy(f)
print('has_flat=', dm_ai_module.has_flat_batch_inference_registered())

s = dm_ai_module.GameState(0)
s.setup_test_duel()
ne = dm_ai_module.NeuralEvaluator({})
policies, values = ne.evaluate([s])
print('done, policies_len=', len(policies))
