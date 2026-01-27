import time
import numpy as np
import onnxruntime as ort
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'models/duel_transformer_20260123_143241.onnx'
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
THREADS = [1, 2, 4]
WARMUP = 10
ITER = 100
SEQ_LEN = 200

print('Model:', MODEL)
for threads in THREADS:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    # Force CPU provider for deterministic measurement
    sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    print(f"\nProvider: CPUExecutionProvider, threads={threads}, input={input_name}")

    for batch in BATCH_SIZES:
        # random token ids in vocab range (use 0..999)
        x = np.random.randint(0, 1000, size=(batch, SEQ_LEN)).astype(np.int64)
        # Warmup
        for _ in range(WARMUP):
            _ = sess.run(None, {input_name: x})
        # Timed runs
        t0 = time.time()
        for _ in range(ITER):
            _ = sess.run(None, {input_name: x})
        t1 = time.time()
        total_time = t1 - t0
        avg_batch_ms = (total_time / ITER) * 1000.0
        samples_per_sec = (batch * ITER) / total_time
        print(f"batch={batch:2d} avg_batch_ms={avg_batch_ms:7.2f}ms samples/s={samples_per_sec:7.1f}")

print('\nDone')
