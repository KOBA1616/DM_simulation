import importlib.util
import importlib.machinery
import os
import sys
import random
import pytest

HERE = os.path.dirname(__file__)
WRAPPER_PATH = os.path.join(HERE, "index_to_command.py")
# load wrapper
spec = importlib.util.spec_from_file_location("idx_wrapper", WRAPPER_PATH)
idx_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idx_wrapper)

# Load local Python encoder for TOTAL size
from native_prototypes.index_to_command.command_encoder import CommandEncoder, index_to_command as py_index_to_command


def _try_load_built_pyd():
    # Search for built .pyd in common build output locations
    candidates = []
    search_root = os.path.join(HERE, "build")
    for root, dirs, files in os.walk(search_root):
        for f in files:
            if f.startswith("index_to_command_native") and f.endswith(".pyd"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        return None
    # prefer Release if present
    candidates.sort(key=lambda p: ("Release" not in p, p))
    path = candidates[0]
    try:
        name = "index_to_command_native_local"
        loader = importlib.machinery.ExtensionFileLoader(name, path)
        spec = importlib.util.spec_from_loader(name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod
    except Exception:
        return None


native_mod = None
# prefer local built pyd if available
native_mod = _try_load_built_pyd()
if native_mod is None:
    # fallback to import by name if installed
    try:
        import index_to_command_native as native_mod
    except Exception:
        native_mod = None


@pytest.mark.parametrize("count", [100])
def test_parity_random_and_edge(count):
    if native_mod is None:
        pytest.skip("native module not available (built .pyd not found and module not importable)")

    total = CommandEncoder.TOTAL_COMMAND_SIZE
    rng = random.Random(12345)
    # sample edges and random indices
    indices = set([0, 1, 2, total - 1, total - 2, CommandEncoder.PLAY_FROM_ZONE_BASE, CommandEncoder.PLAY_FROM_ZONE_BASE + 1])
    while len(indices) < count:
        indices.add(rng.randrange(0, total))

    for i in sorted(indices):
        py_out = py_index_to_command(i)
        try:
            native_out = native_mod.index_to_command(int(i))
        except Exception as e:
            pytest.fail(f"native index_to_command failed for index {i}: {e}")
        assert py_out == native_out, f"Mismatch at index {i}: py={py_out} native={native_out}"
