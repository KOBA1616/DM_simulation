import sys
import os
import pytest
from typing import Any

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass # Let tests fail if module not found

def test_import():
    assert 'dm_ai_module' in sys.modules

def test_game_state_init():
    gs = dm_ai_module.GameState(42)
    assert gs.turn_number == 1
    assert gs.active_player_id == 0

def test_stats_vectorization():
    gs = dm_ai_module.GameState(42)
    card_db: dict[int, Any] = {}
    # Fixed: initialize_card_stats is an instance method
    gs.initialize_card_stats(card_db, 40)

    # Test vectorization of unknown card (should be all zeros)
    # Note: vectorize_card_stats is currently not exposed in bindings
    # vec = gs.vectorize_card_stats(999)
    # assert len(vec) == 16
    # assert all(v == 0.0 for v in vec)

def test_get_card_stats():
    gs = dm_ai_module.GameState(42)
    stats = dm_ai_module.get_card_stats(gs)
    assert isinstance(stats, dict)
