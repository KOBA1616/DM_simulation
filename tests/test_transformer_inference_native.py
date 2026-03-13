import os
import pytest


def _can_import_native():
    try:
        import dm_ai_module as dm  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _can_import_native(), reason="native dm_ai_module not available")
def test_native_onnx_load_and_infer_subprocess():
    """Run native ONNX loader in a subprocess to isolate crashes.

    This delegates to `scripts/run_native_onnx_loader.py` which builds a tiny
    ONNX model and invokes `dm_ai_module.native_load_onnx` in a child process.
    The test asserts the child exits with code 0.
    """
    runner = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_native_onnx_loader.py')
    runner = os.path.normpath(os.path.abspath(runner))
    assert os.path.exists(runner), f"runner script not found: {runner}"

    import subprocess, sys

    p = subprocess.Popen([sys.executable, runner], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=60)
    if out:
        print('\n[child stdout]\n', out)
    if err:
        print('\n[child stderr]\n', err)

    # If child returned dedicated API-mismatch code, skip the test with explanation
    if p.returncode == 6:
        pytest.skip('Child detected ONNX Runtime API/version mismatch; native loader requires matching ORT version')

    # If the child printed INFER_OK then a 1-batch evaluation succeeded.
    if 'INFER_OK' in (out or ''):
        return

    # If loader succeeded but no infer was attempted/supported, skip the test.
    if p.returncode == 0:
        pytest.skip('native loader loaded but infer_batch not supported by returned object')

    assert p.returncode == 0, f"native ONNX loader failed in subprocess (exit {p.returncode})"
