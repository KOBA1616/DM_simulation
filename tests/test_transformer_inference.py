import os
import tempfile
import numpy as np
import pytest


def _can_import(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_can_import('onnx') and _can_import('onnxruntime')),
                    reason="Requires onnx and onnxruntime packages")
def test_onnx_export_and_inference_basic():
    import onnx
    import onnx.helper as helper
    import onnx.numpy_helper as numpy_helper
    import onnxruntime as ort

    # Build a tiny graph: Y = X * W
    input_dim = 8
    output_dim = 4

    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, ['N', input_dim])
    Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, ['N', output_dim])

    # Weight initializer
    W_np = np.arange(input_dim * output_dim, dtype=np.float32).reshape((input_dim, output_dim))
    W = numpy_helper.from_array(W_np, name='W')

    matmul_node = helper.make_node('MatMul', ['X', 'W'], ['Y'])

    graph = helper.make_graph(
        [matmul_node],
        'tiny_graph',
        [X],
        [Y],
        initializer=[W]
    )

    model = helper.make_model(
        graph,
        producer_name='dm_sim_smoke',
        opset_imports=[helper.make_operatorsetid('', 13)],
    )
    # 再発防止: onnxruntime==1.20.1 環境では高すぎる IR version を読むと失敗するため、
    # スモーク用モデルは互換性が広い IR v10 に固定する。
    model.ir_version = 10

    # Save to temp file
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'tiny.onnx')
        onnx.save_model(model, path)

        # Run with onnxruntime
        sess = ort.InferenceSession(path)
        inp = np.ones((2, input_dim), dtype=np.float32)
        out = sess.run(None, {'X': inp})

        assert len(out) == 1
        y = out[0]
        # Expected: each row is sum over inputs weighted by W
        assert y.shape == (2, output_dim)
        # Basic numeric sanity: not NaN and finite
        assert np.isfinite(y).all()
