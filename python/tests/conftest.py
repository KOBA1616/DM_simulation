import pytest
import sys
import os

# Ensure bin directory is in path
BIN_PATH = os.path.join(os.path.dirname(__file__), "../../bin")
if BIN_PATH not in sys.path:
    sys.path.append(BIN_PATH)

# Try import
try:
    import dm_ai_module
except ImportError:
    print(f"Warning: dm_ai_module not found in {BIN_PATH}")

@pytest.fixture
def card_db():
    # Return a minimal or mocked card DB
    return {}

@pytest.fixture
def game_state(card_db):
    if 'dm_ai_module' not in sys.modules:
        pytest.skip("dm_ai_module not installed")
    # Direct usage of C++ GameState, no wrapper
    return dm_ai_module.GameState(100)
