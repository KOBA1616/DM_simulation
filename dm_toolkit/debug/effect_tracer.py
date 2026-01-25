# -*- coding: utf-8 -*-
"""
Effect Tracer for Debugging Card Effects and Game Flow.

This module provides the EffectTracer class, which hooks into the game execution
to record the sequence of effects, commands, and state changes. It is designed
to assist in debugging complex card interactions and verifying effect resolution logic.

Capabilities:
- Trace execution flow (commands, effects, triggers)
- Record snapshots of game state (optional)
- Export trace to JSON format
"""

import json
import time
from typing import List, Dict, Any, Optional
from enum import Enum

class TraceEventType(Enum):
    COMMAND_EXECUTION = "COMMAND"
    EFFECT_RESOLUTION = "EFFECT"
    START_EFFECT = "START_EFFECT"
    END_EFFECT = "END_EFFECT"
    STEP = "STEP"
    TRIGGER_ACTIVATION = "TRIGGER"
    STATE_CHANGE = "STATE"
    INFO = "INFO"
    ERROR = "ERROR"

class EffectTracer:
    def __init__(self) -> None:
        self._trace_log: List[Dict[str, Any]] = []
        self._enabled = False
        self._start_time = 0.0

    def start_tracing(self) -> None:
        """Starts recording the trace."""
        self._trace_log = []
        self._enabled = True
        self._start_time = time.time()
        self.log_event(TraceEventType.INFO, "Tracing Started")

    def stop_tracing(self) -> None:
        """Stops recording."""
        self.log_event(TraceEventType.INFO, "Tracing Stopped")
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def log_event(self, event_type: TraceEventType, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Logs an event to the trace.

        Args:
            event_type: The category of the event.
            message: A human-readable description.
            data: Additional context data (e.g. command parameters, card ID).
        """
        if not self._enabled:
            return

        timestamp = time.time() - self._start_time
        entry = {
            "timestamp": round(timestamp, 4),
            "type": event_type.value,
            "message": message,
            "data": data or {}
        }
        self._trace_log.append(entry)

    def log_command(self, command: Dict[str, Any]) -> None:
        """Helper to log a command execution."""
        cmd_type = command.get("type", "UNKNOWN")
        self.log_event(TraceEventType.COMMAND_EXECUTION, f"Executing {cmd_type}", command)

    def log_state_snapshot(self, state: Any) -> None:
        """
        Logs a snapshot of the game state.
        Note: This can be heavy, use sparingly or filter essential data.
        """
        if not self._enabled:
            return

        # Create a simplified snapshot
        try:
            snapshot = {
                "turn": getattr(state, "turn_number", -1),
                "phase": getattr(state, "current_phase", -1),
                "active_player": getattr(state, "active_player_id", -1),
                "pending_effects_count": len(getattr(state, "pending_effects", []))
            }
            self.log_event(TraceEventType.STATE_CHANGE, "State Snapshot", snapshot)
        except Exception as e:
            self.log_event(TraceEventType.INFO, f"Failed to snapshot state: {e}")

    def export_to_json(self, filepath: str) -> None:
        """Exports the recorded trace to a JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._trace_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error exporting trace to {filepath}: {e}")

    def export_for_flowchart(self) -> Dict[str, Any]:
        """
        Exports data in a format suitable for flowchart generation.
        Returns a dictionary representing nodes and edges or steps.
        """
        steps = []
        depth = 0
        for entry in self._trace_log:
            # Create a copy to avoid modifying the original log if we needed to
            item = entry.copy()
            event_type = entry.get("type")

            if event_type == TraceEventType.START_EFFECT.value:
                item['depth'] = depth
                depth += 1
            elif event_type == TraceEventType.END_EFFECT.value:
                depth = max(0, depth - 1)
                item['depth'] = depth
            else:
                item['depth'] = depth
            steps.append(item)

        return {
            "trace_id": str(time.time()),
            "steps": steps
        }

    def start_effect(self, effect_name: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.log_event(TraceEventType.START_EFFECT, f"Start effect: {effect_name}", context)

    def end_effect(self, effect_name: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.log_event(TraceEventType.END_EFFECT, f"End effect: {effect_name}", context)

    def step(self, description: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.log_event(TraceEventType.STEP, description, details)

    def error(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.log_event(TraceEventType.ERROR, message, details)

    def get_trace(self) -> List[Dict[str, Any]]:
        return self._trace_log

    def clear(self) -> None:
        self._trace_log = []

# Global instance for easy access if needed (singleton pattern)
_tracer_instance = EffectTracer()

def get_tracer() -> EffectTracer:
    return _tracer_instance
