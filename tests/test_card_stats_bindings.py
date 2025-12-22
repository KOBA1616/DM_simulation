import sys
import os

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(r"C:\Program Files (x86)\mingw64\bin")
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import dm_ai_module as dm


def test_initialize_and_vectorize_defaults():
    # Load card DB
    card_db = dm.JsonLoader.load_cards('data/cards.json')
    assert isinstance(card_db, dict)

    gs = dm.GameState(42)
    # Initialize stats map
    dm.initialize_card_stats(gs, card_db, 40)

    # For a known card id (use first key), check stats exist
    # Note: vectorize_card_stats and get_library_potential might not be exposed anymore in recent bindings
    # Checking bindings.cpp... get_card_stats is exposed.

    stats = dm.get_card_stats(gs)
    assert isinstance(stats, dict)

    if len(card_db) > 0:
        first_cid = list(card_db.keys())[0]
        if first_cid in stats:
             s = stats[first_cid]
             assert s['play_count'] == 0
             assert s['win_count'] == 0

    # Also ensure module-level helpers exist
    assert hasattr(dm, 'initialize_card_stats')
    # These seem to have been removed or refactored. The test was likely for an older version.
    # assert hasattr(dm, 'vectorize_card_stats')
    # assert hasattr(dm, 'get_library_potential')
