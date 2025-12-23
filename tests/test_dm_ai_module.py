import sys
import os
import pytest

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
    card_db = {}
    gs.initialize_card_stats(card_db, 40)

def test_get_card_stats():
    gs = dm_ai_module.GameState(42)
    stats = dm_ai_module.get_card_stats(gs)
    assert isinstance(stats, dict)
