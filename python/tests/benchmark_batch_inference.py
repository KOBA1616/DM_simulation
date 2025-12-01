import time
import numpy as np
import dm_ai_module


def make_states(batch):
    states = [dm_ai_module.GameState(i) for i in range(batch)]
    for s in states:
        s.setup_test_duel()
    return states


def run_eval(ne, states, iters=100):
    # Warmup
    for _ in range(10):
        ne.evaluate(states)
    t0 = time.perf_counter()
    for _ in range(iters):
        ne.evaluate(states)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def bench_numpy(batch, iters=200):
    print(f"\nBenchmark numpy flat path: batch={batch}")
    def model(arr: np.ndarray):
        # simple zero outputs
        return np.zeros((arr.shape[0], dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE), dtype=np.float32), np.zeros((arr.shape[0],), dtype=np.float32)

    dm_ai_module.register_batch_inference_numpy(model)
    ne = dm_ai_module.NeuralEvaluator({})
    states = make_states(batch)
    avg = run_eval(ne, states, iters=iters)
    print(f"numpy avg latency: {avg*1000:.3f} ms per evaluate()")


def bench_list(batch, iters=200):
    print(f"\nBenchmark list-of-lists path: batch={batch}")

    def model(lst):
        # lst is list[list[float]]
        batch = len(lst)
        return [[0.0]*dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE for _ in range(batch)], [0.0]*batch

    dm_ai_module.register_batch_inference(model)
    ne = dm_ai_module.NeuralEvaluator({})
    states = make_states(batch)
    avg = run_eval(ne, states, iters=iters)
    print(f"list-of-lists avg latency: {avg*1000:.3f} ms per evaluate()")


def main():
    batches = [1, 8, 32, 128]
    for b in batches:
        bench_numpy(b, iters=200 if b<=32 else 100)
        bench_list(b, iters=200 if b<=32 else 100)


if __name__ == '__main__':
    main()
