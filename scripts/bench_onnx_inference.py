import time
import numpy as np
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'models/duel_transformer_20260123_143241.onnx'
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
THREADS = [1, 2, 4]
WARMUP = 10
ITER = 100
SEQ_LEN = 200

# Prefer native C++ OnnxModel binding when available; fall back to onnxruntime Python
use_native = False
native_model = None
try:
    import dm_ai_module
    if hasattr(dm_ai_module, 'native_load_onnx'):
        ok = dm_ai_module.native_load_onnx(MODEL)
        if not ok:
            print(f"Failed to load native ONNX model: {MODEL}")
            sys.exit(1)
        use_native = True
    else:
        use_native = False
except Exception:
    use_native = False

if not use_native:
    import onnxruntime as ort

print('Model:', MODEL)
for threads in THREADS:
    if use_native:
        print(f"\nProvider: native-C++ OnnxModel, threads={threads}")
    else:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads
        # Force CPU provider for deterministic measurement
        sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        print(f"\nProvider: CPUExecutionProvider, threads={threads}, input={input_name}")

    for batch in BATCH_SIZES:
        # random token ids in vocab range (use 0..999)
        x = np.random.randint(0, 1000, size=(batch, SEQ_LEN))

        if use_native:
            # Native binding: convert tokens to float and flatten to match native_infer_flat(flat, batch_size, input_size)
            xf = x.astype(np.float32).ravel()
            # Warmup
            for _ in range(WARMUP):
                _ = dm_ai_module.native_infer_flat(xf, batch, SEQ_LEN)

            t0 = time.time()
            for _ in range(ITER):
                _ = dm_ai_module.native_infer_flat(xf, batch, SEQ_LEN)
            t1 = time.time()
        else:
            x = x.astype(np.int64)
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
