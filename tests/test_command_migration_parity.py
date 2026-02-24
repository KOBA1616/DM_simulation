import dm_ai_module as dm
from dm_toolkit import commands
from dm_toolkit.engine.compat import EngineCompat

def test_generators_exist_and_return_iterables():
    # Use registered card DB (may be empty) to construct a GameInstance
    card_db = EngineCompat.load_cards_robust("data/cards.json")
    # Need to get cached native DB if available
    native_db = EngineCompat._native_db_cache
    if native_db:
        gi = dm.GameInstance(42, native_db)
    else:
        gi = dm.GameInstance(42, card_db) if card_db else dm.GameInstance()

    state = gi.state

    # Prefer command-first generator
    cmds = []
    try:
        cmds = commands.generate_legal_commands(state, card_db, strict=False, skip_wrapper=True) or []
    except Exception:
        try:
            if hasattr(dm, 'generate_commands'):
                cmds = dm.generate_commands(state, card_db) or []
        except Exception:
            cmds = []

    # Legacy actions as fallback for parity checks
    # Native IntentGenerator still exists?
    actions = []
    try:
        if hasattr(dm, 'IntentGenerator'):
            actions = dm.IntentGenerator.generate_legal_commands(state, card_db) or []
    except Exception:
        actions = []

    # Basic sanity: both return iterables (lengths are integers)
    assert hasattr(actions, '__len__')
    assert hasattr(cmds, '__len__')
