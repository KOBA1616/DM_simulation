from types import SimpleNamespace
from dm_toolkit.engine import compat
from dm_toolkit.engine.compat import EngineCompat


def test_enginecompat_calls_native_ces_recalculate(monkeypatch):
    called = {'v': False}

    class CES:
        @staticmethod
        def recalculate(native_state):
            called['v'] = True

    dummy_mod = SimpleNamespace(ContinuousEffectSystem=CES)

    # Inject dummy native module into compat
    monkeypatch.setattr(compat, 'dm_ai_module', dummy_mod, raising=False)

    class NativeState:
        pass

    native_state = NativeState()

    class State:
        def __init__(self):
            self._native = native_state

    s = State()

    EngineCompat.ensure_recalculated(s)

    assert called['v'] is True
