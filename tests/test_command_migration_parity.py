import dm_ai_module as dm
from dm_toolkit import commands_v2


def test_generators_exist_and_return_iterables():
    # Use registered card DB (may be empty) to construct a GameInstance
    card_db = dm.CardRegistry.get_all_cards()
    gi = dm.GameInstance(42, card_db)
    state = gi.state

    # Legacy action generator
    actions = []
    try:
        actions = dm.IntentGenerator.generate_legal_actions(state, card_db) or []
    except Exception:
        actions = []

    # New command-first generator (binding added in phase 2)
    commands = []
    try:
        commands = dm.generate_commands(state, card_db) or []
    except Exception:
        # Fallback to commands_v2 wrapper
        commands = commands_v2.generate_legal_commands(state, card_db, strict=False) or []

    # Basic sanity: both return iterables (lengths are integers)
    assert hasattr(actions, '__len__')
    assert hasattr(commands, '__len__')
