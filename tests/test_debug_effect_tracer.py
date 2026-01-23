import pytest
import json
import time
import os
from dm_toolkit.debug.effect_tracer import EffectTracer, TraceEventType, get_tracer

class TestDebugEffectTracer:

    @pytest.fixture
    def tracer(self):
        """Fixture to provide a clean tracer instance for each test."""
        tracer = EffectTracer()
        # Reset state just in case, though it's a new instance
        tracer._enabled = False
        tracer._trace_log = []
        return tracer

    def test_initial_state(self, tracer):
        assert not tracer.is_enabled()
        assert tracer.get_trace() == []

    def test_start_stop_tracing(self, tracer):
        tracer.start_tracing()
        assert tracer.is_enabled()
        trace = tracer.get_trace()
        assert len(trace) == 1
        assert trace[0]["type"] == TraceEventType.INFO.value
        assert trace[0]["message"] == "Tracing Started"

        tracer.stop_tracing()
        assert not tracer.is_enabled()
        trace = tracer.get_trace()
        assert len(trace) == 2
        assert trace[-1]["type"] == TraceEventType.INFO.value
        assert trace[-1]["message"] == "Tracing Stopped"

    def test_log_event(self, tracer):
        tracer.start_tracing()
        tracer.log_event(TraceEventType.COMMAND_EXECUTION, "Test Command", {"id": 1})

        trace = tracer.get_trace()
        # 0: Started, 1: Logged Event
        assert len(trace) == 2
        event = trace[1]
        assert event["type"] == TraceEventType.COMMAND_EXECUTION.value
        assert event["message"] == "Test Command"
        assert event["data"] == {"id": 1}
        assert "timestamp" in event
        assert isinstance(event["timestamp"], float)

    def test_log_when_disabled(self, tracer):
        tracer.log_event(TraceEventType.INFO, "Should not be logged")
        assert len(tracer.get_trace()) == 0

    def test_log_command(self, tracer):
        tracer.start_tracing()
        cmd = {"type": "PLAY_CARD", "card_id": 123}
        tracer.log_command(cmd)

        trace = tracer.get_trace()
        event = trace[-1]
        assert event["type"] == TraceEventType.COMMAND_EXECUTION.value
        assert event["message"] == "Executing PLAY_CARD"
        assert event["data"] == cmd

    def test_log_state_snapshot(self, tracer):
        class MockState:
            turn_number = 5
            current_phase = 2
            active_player_id = 1
            pending_effects = ["effect1", "effect2"]

        tracer.start_tracing()
        state = MockState()
        tracer.log_state_snapshot(state)

        trace = tracer.get_trace()
        event = trace[-1]
        assert event["type"] == TraceEventType.STATE_CHANGE.value
        assert event["message"] == "State Snapshot"
        assert event["data"]["turn"] == 5
        assert event["data"]["phase"] == 2
        assert event["data"]["active_player"] == 1
        assert event["data"]["pending_effects_count"] == 2

    def test_log_state_snapshot_error(self, tracer):
        # Test error handling when state causes an exception
        class BadState:
             # This will cause len() to fail because int has no len()
             pending_effects = 123

        tracer.start_tracing()
        tracer.log_state_snapshot(BadState())

        trace = tracer.get_trace()
        event = trace[-1]
        # Should log INFO with error message
        assert event["type"] == TraceEventType.INFO.value
        assert "Failed to snapshot state" in event["message"]

    def test_export_to_json(self, tracer, tmp_path):
        tracer.start_tracing()
        tracer.log_event(TraceEventType.INFO, "Export Test")
        tracer.stop_tracing()

        export_path = tmp_path / "trace.json"
        tracer.export_to_json(str(export_path))

        assert os.path.exists(export_path)
        with open(export_path, 'r') as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) >= 2 # Start, Export Test, Stop

    def test_singleton_accessor(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2
        assert isinstance(t1, EffectTracer)

    def test_clear(self, tracer):
        tracer.start_tracing()
        tracer.log_event(TraceEventType.INFO, "Msg")
        assert len(tracer.get_trace()) > 0
        tracer.clear()
        assert len(tracer.get_trace()) == 0

    def test_flowchart_logic(self, tracer):
        tracer.start_tracing()
        tracer.start_effect("Root")
        tracer.step("Step 1")
        tracer.start_effect("Nested")
        tracer.end_effect("Nested")
        tracer.end_effect("Root")

        flow = tracer.export_for_flowchart()
        steps = flow["steps"]
        assert len(steps) == 6 # Tracing Started + 5 events

        # 0: Tracing Started (INFO)
        # 1: Start Root
        # 2: Step 1
        # 3: Start Nested
        # 4: End Nested
        # 5: End Root

        # Verify Depths
        # INFO: depth 0
        assert steps[0]["depth"] == 0
        # Start Root: depth 0, then depth becomes 1
        assert steps[1]["type"] == TraceEventType.START_EFFECT.value
        assert steps[1]["depth"] == 0
        # Step 1: depth 1
        assert steps[2]["depth"] == 1
        # Start Nested: depth 1, then depth becomes 2
        assert steps[3]["depth"] == 1
        # End Nested: depth becomes 1, then item gets depth 1
        assert steps[4]["type"] == TraceEventType.END_EFFECT.value
        assert steps[4]["depth"] == 1
        # End Root: depth becomes 0, then item gets depth 0
        assert steps[5]["depth"] == 0

    def test_helper_methods(self, tracer):
        tracer.start_tracing()
        tracer.error("Fail", {"code": 500})
        trace = tracer.get_trace()
        assert trace[-1]["type"] == TraceEventType.ERROR.value
        assert trace[-1]["message"] == "Fail"
