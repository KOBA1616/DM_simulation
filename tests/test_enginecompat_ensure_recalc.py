from dm_toolkit.engine.compat import EngineCompat


class DummyState:
    def __init__(self):
        self.called = False

    def ensure_recalculated(self):
        self.called = True


def test_enginecompat_calls_ensure_recalculated():
    s = DummyState()
    EngineCompat.ensure_recalculated(s)
    assert s.called is True
