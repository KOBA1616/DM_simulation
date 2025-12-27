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
    Serialized to a flat dictionary matching the Command Schema.
    """

    uid: str = field(default_factory=_new_id)
    type: str = "UNKNOWN"

    # Specific known fields for explicit handling
    input_value_key: Optional[str] = None
    output_value_key: Optional[str] = None

    # Recursive structures (lists of commands)
    if_true: Optional[List[Any]] = None
    if_false: Optional[List[Any]] = None
    options: Optional[List[List[Any]]] = None
    on_error: Optional[List[Any]] = None

    # Catch-all for other fields (amount, str_param, target_filter, etc.)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Start with params to form the base (flattened)
        d = self.params.copy()

        # Overwrite/Set explicit fields
        d["uid"] = self.uid
        d["type"] = self.type

        if self.input_value_key:
            d["input_value_key"] = self.input_value_key
        if self.output_value_key:
            d["output_value_key"] = self.output_value_key

        if self.if_true:
            d["if_true"] = self.if_true
        if self.if_false:
            d["if_false"] = self.if_false
        if self.options:
            d["options"] = self.options
        if self.on_error:
            d["on_error"] = self.on_error

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommandDef":
        # Extract known fields
        uid = d.get("uid", d.get("id", _new_id())) # Fallback to id if uid missing
        ctype = d.get("type", "UNKNOWN")
        input_key = d.get("input_value_key")
        # Legacy support for input_keys list
        if not input_key and d.get("input_keys"):
            val = d["input_keys"]
            if isinstance(val, list) and val:
                input_key = val[0]

        output_key = d.get("output_value_key")
        if not output_key and d.get("output_keys"):
            val = d["output_keys"]
            if isinstance(val, list) and val:
                output_key = val[0]

        if_true = d.get("if_true")
        if_false = d.get("if_false")
        options = d.get("options")
        on_error = d.get("on_error")

        # Gather remainder into params
        known_keys = {
            "uid", "id", "type",
            "input_value_key", "input_keys",
            "output_value_key", "output_keys",
            "if_true", "if_false", "options", "on_error"
        }
        params = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            uid=uid,
            type=ctype,
            input_value_key=input_key,
            output_value_key=output_key,
            if_true=if_true,
            if_false=if_false,
            options=options,
            on_error=on_error,
            params=params
        )


@dataclass
class WarningCommand(CommandDef):
    """Special node for unconverted legacy actions.

    The GUI displays this as a warning.
    """

    warning: str = "Unconverted legacy action"
    original_action: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["legacy_warning"] = True
        base["warning"] = self.warning
        if self.original_action:
            base["legacy_original_action"] = self.original_action
        base["_warning_command"] = True
        return base

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WarningCommand":
        # Extract base fields using same logic as CommandDef
        uid = d.get("uid", d.get("id", _new_id()))
        ctype = d.get("type", "WARNING")
        input_key = d.get("input_value_key")
        output_key = d.get("output_value_key")
        if_true = d.get("if_true")
        if_false = d.get("if_false")
        options = d.get("options")
        on_error = d.get("on_error")

        known_keys = {
            "uid", "id", "type",
            "input_value_key", "input_keys",
            "output_value_key", "output_keys",
            "if_true", "if_false", "options", "on_error",
            "warning", "legacy_warning", "original_action", "legacy_original_action", "_warning_command"
        }
        params = {k: v for k, v in d.items() if k not in known_keys}

        obj = cls(
            uid=uid,
            type=ctype,
            input_value_key=input_key,
            output_value_key=output_key,
            if_true=if_true,
            if_false=if_false,
            options=options,
            on_error=on_error,
            params=params,
            warning=d.get("warning", "Unconverted legacy action"),
            original_action=d.get("original_action") or d.get("legacy_original_action")
        )
        return obj


__all__ = ["CommandDef", "WarningCommand"]
