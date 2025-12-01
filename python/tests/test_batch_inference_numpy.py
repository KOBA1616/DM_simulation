import numpy as np
import dm_ai_module


def sample_numpy_model(arr: np.ndarray):
    # arr: (batch, stride)
    print('sample_numpy_model called, arr.shape=', arr.shape, 'dtype=', arr.dtype)
    batch = arr.shape[0]
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
    policies = np.zeros((batch, action_size), dtype=np.float32)
    values = np.zeros((batch,), dtype=np.float32)
    return policies, values


def run_test():
    print('registering numpy model')
    dm_ai_module.register_batch_inference_numpy(sample_numpy_model)
    print('has_batch_registered (old)=', dm_ai_module.has_batch_inference_registered())
    print('has_flat_registered (numpy)=', dm_ai_module.has_flat_batch_inference_registered())

    s = dm_ai_module.GameState(0)
    s.setup_test_duel()

    ne = dm_ai_module.NeuralEvaluator({})
    policies, values = ne.evaluate([s])
    print('policies_len=', len(policies), 'values_len=', len(values))
    if policies:
        print('policy0_len=', len(policies[0]))

    assert len(values) == 1
    assert len(policies) == 1


if __name__ == '__main__':
    run_test()
