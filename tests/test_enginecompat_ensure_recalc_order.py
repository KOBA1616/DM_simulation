from types import ModuleType, SimpleNamespace
import importlib

from dm_toolkit.engine.compat import EngineCompat


def test_ensure_recalculated_calls_state_method():
    class DummyState:
        def __init__(self):
            self.called = False

        def ensure_recalculated(self):
            self.called = True

    s = DummyState()
    EngineCompat.ensure_recalculated(s)
    assert s.called is True


def test_ensure_recalculated_calls_recalculate_fallback():
    class DummyState:
        def __init__(self):
            self.called = False

        def recalculate(self):
            self.called = True

    s = DummyState()
    EngineCompat.ensure_recalculated(s)
    assert s.called is True


def test_ensure_recalculated_prefers_native_ces(monkeypatch):
    called = {}

    class CES:
        @staticmethod
        def recalculate(native_state):
            called['native'] = native_state

    # Monkeypatch dm_ai_module used by EngineCompat
    dummy_mod = ModuleType('dm_ai_module_dummy')
    dummy_mod.ContinuousEffectSystem = CES
    monkeypatch.setattr('dm_toolkit.engine.compat.dm_ai_module', dummy_mod)

    # Create state with a native backing
    s = SimpleNamespace()
    native_obj = object()
    s._native = native_obj

    EngineCompat.ensure_recalculated(s)

    assert 'native' in called and called['native'] is native_obj
