from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional
from enum import Enum, auto

class TraceEventType(Enum):
    START_EFFECT = auto()
    END_EFFECT = auto()
    STEP = auto()
    ERROR = auto()
    INFO = auto()

class EffectTraceEntry:
    def __init__(self, event_type: TraceEventType, description: str, data: Optional[Dict[str, Any]] = None, timestamp: float = 0.0):
        self.event_type = event_type
        self.description = description
        self.data = data or {}
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.name,
            "description": self.description,
            "data": self.data,
            "timestamp": self.timestamp
        }

class EffectTracer:
    """
    Collects history of effect resolution.
    """
    def __init__(self):
        self._history: List[EffectTraceEntry] = []
        self._enabled: bool = True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def clear(self):
        self._history = []

    def record(self, event_type: TraceEventType, description: str, data: Optional[Dict[str, Any]] = None):
        if not self._enabled:
            return
        entry = EffectTraceEntry(event_type, description, data)
        self._history.append(entry)

    def start_effect(self, effect_name: str, context: Optional[Dict[str, Any]] = None):
        self.record(TraceEventType.START_EFFECT, f"Start effect: {effect_name}", context)

    def end_effect(self, effect_name: str, context: Optional[Dict[str, Any]] = None):
        self.record(TraceEventType.END_EFFECT, f"End effect: {effect_name}", context)

    def step(self, description: str, details: Optional[Dict[str, Any]] = None):
        self.record(TraceEventType.STEP, description, details)

    def info(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.record(TraceEventType.INFO, message, details)

    def error(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.record(TraceEventType.ERROR, message, details)

    def get_history(self) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in self._history]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.get_history(), indent=indent)

    def save_json(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    # Integration for HTML/Flowchart generators
    def export_for_flowchart(self) -> Dict[str, Any]:
        """
        Exports data in a format suitable for flowchart generation.
        Returns a dictionary representing nodes and edges or steps.
        """
        steps = []
        depth = 0
        for entry in self._history:
            item = entry.to_dict()
            if entry.event_type == TraceEventType.START_EFFECT:
                item['depth'] = depth
                depth += 1
            elif entry.event_type == TraceEventType.END_EFFECT:
                depth = max(0, depth - 1)
                item['depth'] = depth
            else:
                item['depth'] = depth
            steps.append(item)

        return {
            "trace_id": str(time.time()),
            "steps": steps
        }

    # API for GUI display
    def get_trace_summary(self) -> List[str]:
        return [f"[{e.timestamp:.3f}] {e.event_type.name}: {e.description}" for e in self._history]
