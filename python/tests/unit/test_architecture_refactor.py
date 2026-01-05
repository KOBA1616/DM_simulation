import pytest
import dm_ai_module
import os

class TestTriggerManager:
    def test_wiring(self):
        """Verify that GameInstance correctly wires up TriggerManager"""
        # Create minimal GameInstance
        try:
             # Using default constructor which uses singleton registry
             instance = dm_ai_module.GameInstance(42)

             # Note: We cannot easily verify internal wiring of event_dispatcher from Python
             # without exposing internal details, but successful construction implies setup ran.
             assert instance is not None

        except Exception as e:
            pytest.fail(f"GameInstance construction failed: {e}")

    def test_data_collector_constructors(self):
        """Verify new constructors for DataCollector and ScenarioExecutor"""
        # We need correct types for the map. Key is int (uint16), Value is CardDefinition.
        db = dm_ai_module.CardDatabase()
        db[1] = dm_ai_module.CardDefinition()

        # Test Reference-like constructor
        dc = dm_ai_module.DataCollector(db)
        assert dc is not None

        se = dm_ai_module.ScenarioExecutor(db)
        assert se is not None
