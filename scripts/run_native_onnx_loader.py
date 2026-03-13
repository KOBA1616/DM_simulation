#!/usr/bin/env python3
"""Create a tiny ONNX model and attempt to load it via dm_ai_module.native_load_onnx
in a separate process. This script writes the model to a temp file and invokes a
child Python process that imports dm_ai_module and calls native_load_onnx(path).

Exit codes:
 - 0: native_load_onnx returned a truthy value and (optionally) infer_batch succeeded
 - 2: import dm_ai_module failed (native extension missing)
 - 3: native_load_onnx attribute missing
 - 4: native call returned falsy/indicated failure
 - 5: child process crashed (non-zero exit)
"""
import sys
import os
import tempfile
import subprocess
import textwrap

try:
    import onnx
    import onnx.helper as helper
    import onnx.numpy_helper as numpy_helper
    import numpy as np
except Exception as e:
    print("Required Python packages (onnx, numpy) not available:", e)
    sys.exit(1)

# Build tiny model
input_dim = 8
output_dim = 4
X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, ['N', input_dim])
Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, ['N', output_dim])
W_np = np.arange(input_dim * output_dim, dtype=np.float32).reshape((input_dim, output_dim))
W = numpy_helper.from_array(W_np, name='W')
matmul_node = helper.make_node('MatMul', ['X', 'W'], ['Y'])
graph = helper.make_graph([matmul_node], 'tiny_graph', [X], [Y], initializer=[W])
model = helper.make_model(graph, producer_name='dm_native_runner')

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, 'tiny_native.onnx')
    onnx.save_model(model, path)

    # Prepare child script that will import dm_ai_module and call native_load_onnx
    child_code = textwrap.dedent(f"""
    import sys
    try:
        import dm_ai_module as dm
    except Exception as e:
        print('IMPORT_ERROR', e)
        sys.exit(2)
    loader = getattr(dm, 'native_load_onnx', None)
    if loader is None:
        print('NO_LOADER')
        sys.exit(3)
    try:
        res = loader(r'{path.replace('\\','\\\\')}')
    except Exception as e:
        print('CALL_EXCEPTION', e)
        # Let caller detect crash by non-zero exit
        sys.exit(5)
    if not res:
        print('LOAD_FAILED', res)
        sys.exit(4)
    # Optionally try infer_batch if object supports it
    try:
        if hasattr(res, 'infer_batch'):
            import numpy as np
            batch_size = 1
            flat = np.ones((batch_size * {input_dim},), dtype=np.float32)
            out = res.infer_batch(flat, batch_size, {input_dim})
            print('INFER_OK', type(out))
    except Exception as e:
        print('INFER_EXCEPTION', e)
        sys.exit(5)
    print('SUCCESS')
    sys.exit(0)
    """)

    # Run child in separate Python interpreter to isolate crashes
    p = subprocess.Popen([sys.executable, '-c', child_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=30)
    print('Child exit code:', p.returncode)
    if out:
        print('Child stdout:\n', out)
    if err:
        print('Child stderr:\n', err)

    # Detect common ONNX Runtime API/version mismatch visible in stderr
    stderr_lower = (err or '').lower()
    if 'requested api version' in stderr_lower or 'api version' in stderr_lower and 'onnxruntime' in stderr_lower:
        print('Detected ONNX Runtime API/version mismatch in child process stderr.')
        # Use dedicated exit code 6 to indicate API mismatch (not a crash of native code)
        sys.exit(6)

    sys.exit(p.returncode if p.returncode is not None else 5)
