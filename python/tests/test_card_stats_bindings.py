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
    # Load card DB (CSV loader returns a map-like dict)
    card_db = dm.CsvLoader.load_cards('data/cards.csv')
    assert isinstance(card_db, dict)

    gs = dm.GameState(42)
    # Initialize stats map
    dm.initialize_card_stats(gs, card_db, 40)

    # For a known card id (use first key), vector should be length 16 and zeros (no historical stats loaded)
    first_cid = list(card_db.keys())[0]
    vec = dm.vectorize_card_stats(gs, first_cid)
    assert isinstance(vec, list)
    assert len(vec) == 16
    assert all(v == 0.0 for v in vec)

    # Library potential should also be zeros since initial sums are zero
    pot = dm.get_library_potential(gs)
    assert isinstance(pot, list)
    assert len(pot) == 16
    assert all(p == 0.0 for p in pot)

    # Also ensure module-level helpers exist
    assert hasattr(dm, 'initialize_card_stats')
    assert hasattr(dm, 'vectorize_card_stats')
    assert hasattr(dm, 'get_library_potential')
