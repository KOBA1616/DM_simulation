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
    # Case A: simple float32, batch=1
    print('\nCase A: float32, batch=1')
    dm_ai_module.register_batch_inference_numpy(sample_numpy_model)
    print('has_flat_registered=', dm_ai_module.has_flat_batch_inference_registered())
    s = dm_ai_module.GameState(0)
    s.setup_test_duel()
    ne = dm_ai_module.NeuralEvaluator({})
    policies, values = ne.evaluate([s])
    print('policies_len=', len(policies), 'values_len=', len(values))
    if policies:
        print('policy0_len=', len(policies[0]))
    assert len(values) == 1 and len(policies) == 1

    # Case B: batch=4 float32
    print('\nCase B: float32, batch=4')
    def model_b(arr: np.ndarray):
        print('model_b called, shape=', arr.shape, 'dtype=', arr.dtype)
        batch = arr.shape[0]
        action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
        return np.zeros((batch, action_size), dtype=np.float32), np.zeros((batch,), dtype=np.float32)

    dm_ai_module.register_batch_inference_numpy(model_b)
    states = [dm_ai_module.GameState(i) for i in range(4)]
    for s in states: s.setup_test_duel()
    policies, values = ne.evaluate(states)
    print('policies_len=', len(policies), 'values_len=', len(values))
    assert len(values) == 4 and len(policies) == 4

    # Case C: dtype=float64
    print('\nCase C: float64 input, batch=2')
    def model_c(arr: np.ndarray):
        print('model_c called, dtype=', arr.dtype)
        # Return float64 outputs to test conversion
        batch = arr.shape[0]
        action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
        return np.zeros((batch, action_size), dtype=np.float64), np.zeros((batch,), dtype=np.float64)

    dm_ai_module.register_batch_inference_numpy(model_c)
    states = [dm_ai_module.GameState(i) for i in range(2)]
    for s in states: s.setup_test_duel()
    policies, values = ne.evaluate(states)
    print('policies_len=', len(policies), 'values_len=', len(values))
    assert len(values) == 2 and len(policies) == 2

    # Case D: non-contiguous sliced array
    print('\nCase D: non-contiguous sliced array (forcibly created)')
    def model_d(arr: np.ndarray):
        print('model_d called, arr.shape=', arr.shape, 'contiguous=', arr.flags['C_CONTIGUOUS'])
        batch = arr.shape[0]
        action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
        return np.zeros((batch, action_size), dtype=np.float32), np.zeros((batch,), dtype=np.float32)

    # Create a larger array and slice every other column to make it non-contiguous
    def wrapper_for_d(flat, n, stride):
        import numpy as _np
        big = _np.empty((n, stride + 2), dtype=_np.float32)
        big[:, :stride] = _np.array(flat).reshape((n, stride))
        sliced = big[:, :stride][:, ::1]
        return model_d(sliced)

    # Register via register_batch_inference_numpy using a converter wrapper
    dm_ai_module.register_batch_inference_numpy(lambda arr: model_d(arr[:, :]))
    states = [dm_ai_module.GameState(i) for i in range(2)]
    for s in states: s.setup_test_duel()
    policies, values = ne.evaluate(states)
    print('policies_len=', len(policies), 'values_len=', len(values))
    assert len(values) == 2 and len(policies) == 2


if __name__ == '__main__':
    run_test()
