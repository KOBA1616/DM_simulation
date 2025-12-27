from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class CommandDef:
    """Lightweight Command Definition Model.

    Represents a node in the execution graph.
    Serialized as a flat dictionary to match the engine schema.
    """

    id: str = field(default_factory=_new_id)
    type: str = "UNKNOWN"
    params: Dict[str, Any] = field(default_factory=dict)
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    if_true: Optional[List[Dict[str, Any]]] = None
    if_false: Optional[List[Dict[str, Any]]] = None
    on_error: Optional[str] = None
    options: Optional[List[List[Dict[str, Any]]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a flat dictionary representation matching the storage schema."""
        d = self.params.copy()
        d.update({
            "uid": self.id,
            "type": self.type,
        })

        if self.input_keys:
            d['input_value_key'] = self.input_keys[0]

        if self.output_keys:
            d['output_value_key'] = self.output_keys[0]

        if self.if_true: d['if_true'] = self.if_true
        if self.if_false: d['if_false'] = self.if_false
        if self.options: d['options'] = self.options
        if self.on_error: d['on_error'] = self.on_error

        # Clean up empty optional fields if necessary, or leave them if meaningful.
        # Flat schema typically omits keys that are not present.
        return {k: v for k, v in d.items() if v not in (None, "")}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommandDef":
        """Creates a CommandDef from a flat dictionary (or legacy nested params)."""
        uid = d.get("uid") or d.get("id") or _new_id()
        ctype = d.get("type", "UNKNOWN")

        input_keys = []
        if d.get('input_value_key'): input_keys.append(d['input_value_key'])
        if d.get('input_keys'): input_keys.extend(d['input_keys'])

        output_keys = []
        if d.get('output_value_key'): output_keys.append(d['output_value_key'])
        if d.get('output_keys'): output_keys.extend(d['output_keys'])

        if_true = d.get("if_true")
        if_false = d.get("if_false")
        options = d.get("options")
        on_error = d.get("on_error")

        # Extract params: merge 'params' dict (legacy) with flat keys
        known_keys = {
            'uid', 'id', 'type',
            'input_value_key', 'input_keys',
            'output_value_key', 'output_keys',
            'if_true', 'if_false', 'options', 'on_error',
            'params', '_warning_command', 'warning', 'original_action', 'legacy_warning'
        }

        params = d.get('params', {}).copy()
        for k, v in d.items():
            if k not in known_keys:
                params[k] = v

        return cls(
            id=uid,
            type=ctype,
            params=params,
            input_keys=input_keys,
            output_keys=output_keys,
            if_true=if_true,
            if_false=if_false,
            on_error=on_error,
            options=options
        )


@dataclass
class WarningCommand(CommandDef):
    """Special node representing an unconverted legacy Action.
    """

    warning: str = "Unconverted legacy action"
    original_action: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "legacy_warning": True,
            "warning": self.warning,
            "original_action": self.original_action,
            "type": self.type # ensure type is preserved/overwritten correctly
        })
        return base

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WarningCommand":
        # First create base to handle params extraction
        base = super().from_dict(d)
        return cls(
            id=base.id,
            type=base.type,
            params=base.params,
            input_keys=base.input_keys,
            output_keys=base.output_keys,
            if_true=base.if_true,
            if_false=base.if_false,
            on_error=base.on_error,
            options=base.options,
            warning=d.get("warning", "Unconverted legacy action"),
            original_action=d.get("original_action")
        )


__all__ = ["CommandDef", "WarningCommand"]
