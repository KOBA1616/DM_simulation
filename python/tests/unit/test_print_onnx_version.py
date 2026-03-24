def test_print_onnx():
    import onnxruntime
    print('ORT_VERSION_AT_RUNTIME:', onnxruntime.__version__)
    assert onnxruntime.__version__ == '1.20.1'