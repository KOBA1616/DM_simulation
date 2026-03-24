from dm_toolkit import commands


class DummyState:
    def __init__(self):
        self.recalc_called = False

    def ensure_recalculated(self):
        self.recalc_called = True


def test_generate_legal_triggers_ensure_recalc():
    state = DummyState()
    # card_db can be minimal for this test
    card_db = {}

    # Call generator; it should attempt to call ensure_recalculated
    cmds = commands.generate_legal_commands(state, card_db, strict=False, skip_wrapper=True)

    assert state.recalc_called is True
    # No native actions expected in this environment, so cmds should be list-like
    assert isinstance(cmds, list)
