import importlib.util
import os
import sys
import pytest

HERE = os.path.dirname(__file__)
PY_ENCODER = os.path.join(HERE, 'command_encoder.py')
spec = importlib.util.spec_from_file_location('py_enc', PY_ENCODER)
py_enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(py_enc)

# Try to load native command_encoder_native
native = None
sys.path.insert(0, os.path.join(HERE, 'build_venv_cmdenc', 'Release'))
try:
    import command_encoder_native as native
except Exception:
    native = None


def test_round_trip_python():
    from native_prototypes.index_to_command.command_encoder import CommandEncoder
    for i in [0, 1, 5, 19, 20, 21, CommandEncoder.TOTAL_COMMAND_SIZE - 1]:
        cmd = py_enc.index_to_command(i)
        idx2 = py_enc.command_to_index(cmd)
        assert idx2 == i


@pytest.mark.skipif(native is None, reason='native not built')
def test_native_matches_python_for_samples():
    samples = [0, 1, 5, 19, 20, 21, 50, 100]
    for i in samples:
        py_out = py_enc.index_to_command(i)
        native_out = native.index_to_command(int(i))
        assert py_out == native_out

        # round-trip native -> command -> index
        idx2 = native.command_to_index(native_out)
        assert idx2 == i
