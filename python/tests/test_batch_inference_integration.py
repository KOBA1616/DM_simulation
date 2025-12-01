import pytest
import numpy as np

try:
    import dm_ai_module
except Exception:
    dm_ai_module = None


def make_states(batch):
    states = [dm_ai_module.GameState(i) for i in range(batch)]
    for s in states:
        s.setup_test_duel()
    return states


@pytest.mark.skipif(dm_ai_module is None, reason="dm_ai_module extension not available")
def test_numpy_and_list_return_same_shapes():
    # Prepare models
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

    def numpy_model(arr: np.ndarray):
        batch = arr.shape[0]
        policies = np.zeros((batch, action_size), dtype=np.float32)
        values = np.zeros((batch,), dtype=np.float32)
        return policies, values

    def list_model(lst):
        batch = len(lst)
        policies = [[0.0] * action_size for _ in range(batch)]
        values = [0.0 for _ in range(batch)]
        return policies, values

    # Test for several batch sizes
    for batch in (1, 4, 8):
        states = make_states(batch)

        # NumPy path
        dm_ai_module.register_batch_inference_numpy(numpy_model)
        ne = dm_ai_module.NeuralEvaluator({})
        policies_np, values_np = ne.evaluate(states)
        dm_ai_module.clear_batch_inference_numpy()

        # List path
        dm_ai_module.register_batch_inference(list_model)
        ne2 = dm_ai_module.NeuralEvaluator({})
        policies_ls, values_ls = ne2.evaluate(states)
        dm_ai_module.clear_batch_inference()

        assert len(values_np) == len(values_ls) == batch
        assert len(policies_np) == len(policies_ls) == batch


@pytest.mark.skipif(dm_ai_module is None, reason="dm_ai_module extension not available")
def test_clear_unregisters_callbacks():
    def dummy(arr):
        action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
        return np.zeros((arr.shape[0], action_size), dtype=np.float32), np.zeros((arr.shape[0],), dtype=np.float32)

    dm_ai_module.register_batch_inference_numpy(dummy)
    assert dm_ai_module.has_flat_batch_inference_registered()
    dm_ai_module.clear_batch_inference_numpy()
    assert not dm_ai_module.has_flat_batch_inference_registered()

    def dummy_list(lst):
        action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
        return [[0.0] * action_size for _ in lst], [0.0 for _ in lst]

    dm_ai_module.register_batch_inference(dummy_list)
    assert dm_ai_module.has_batch_inference_registered()
    dm_ai_module.clear_batch_inference()
    assert not dm_ai_module.has_batch_inference_registered()
