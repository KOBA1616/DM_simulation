def test_dump_onnx():
    import onnxruntime
    print('onnx ver', onnxruntime.__version__)
    print('onnx file', getattr(onnxruntime, '__file__', None))
    assert True
