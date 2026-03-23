from dm_toolkit.commands import generate_legal_commands


def test_generate_legal_commands_ensures_recalc_called():
    called = {'v': False}

    class DummyState:
        def ensure_recalculated(self):
            called['v'] = True

    s = DummyState()

    # Should not raise even if native module is absent; ensure_recalculated must be invoked
    res = generate_legal_commands(s, {})

    assert called['v'] is True
