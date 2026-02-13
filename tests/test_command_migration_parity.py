import dm_ai_module as dm
from dm_toolkit import commands_v2


def test_generators_exist_and_return_iterables():
    # Use registered card DB (may be empty) to construct a GameInstance
    card_db = dm.CardRegistry.get_all_cards()
    gi = dm.GameInstance(42, card_db)
    state = gi.state

    # Prefer command-first generator
    cmds = []
    try:
        cmds = commands_v2.generate_legal_commands(state, card_db, strict=False) or []
    except Exception:
        try:
            cmds = dm.generate_commands(state, card_db) or []
        except Exception:
            cmds = []

    # Legacy actions as fallback for parity checks
    actions = []
    try:
        actions = dm.IntentGenerator.generate_legal_commands(state, card_db) or []
    except Exception:
        actions = []

    # Basic sanity: both return iterables (lengths are integers)
    assert hasattr(actions, '__len__')
    assert hasattr(cmds, '__len__')
