import importlib.util
import os
import sys
import pytest

# Load the wrapper module by path so tests don't rely on package imports
HERE = os.path.dirname(__file__)
WRAPPER_PATH = os.path.join(HERE, "index_to_command.py")
spec = importlib.util.spec_from_file_location("idx_wrapper", WRAPPER_PATH)
idx_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idx_wrapper)

# Access the pure-Python fallback function
py_fallback = getattr(idx_wrapper, "_py_index_to_command", None)
if py_fallback is None:
    pytest.skip("No Python fallback available in wrapper", allow_module_level=True)

# Try to import native module; skip test if not present
try:
    import index_to_command_native as native_mod  # type: ignore
    native_available = True
except Exception:
    native_available = False

SAMPLE_INDICES = [0, 1, 5, 19, 20, 21, 50, 100]


def test_python_fallback_round_trip_small():
    # For each sample, ensure the fallback returns a dict with expected keys
    for i in SAMPLE_INDICES:
        out = py_fallback(i)
        assert isinstance(out, dict)
        assert "type" in out


@pytest.mark.skipif(not native_available, reason="native module not available")
def test_native_equals_python_fallback_for_samples():
    for i in SAMPLE_INDICES:
        py_out = py_fallback(i)
        native_out = native_mod.index_to_command(int(i))
        assert py_out == native_out, f"Mismatch at index {i}: py={py_out} native={native_out}"
