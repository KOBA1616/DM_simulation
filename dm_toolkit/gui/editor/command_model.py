from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class CommandDef:
    """軽量のコマンド定義モデル。

    実行グラフ内のノードを表現します。必要に応じてフィールドを拡張してください。
    """

    id: str = field(default_factory=_new_id)
    type: str = "UNKNOWN"
    params: Dict[str, Any] = field(default_factory=dict)
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    if_true: Optional[str] = None
    if_false: Optional[str] = None
    on_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "params": self.params,
            "input_keys": list(self.input_keys),
            "output_keys": list(self.output_keys),
            "if_true": self.if_true,
            "if_false": self.if_false,
            "on_error": self.on_error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommandDef":
        return cls(
            id=d.get("id", _new_id()),
            type=d.get("type", "UNKNOWN"),
            params=d.get("params", {}),
            input_keys=d.get("input_keys", []),
            output_keys=d.get("output_keys", []),
            if_true=d.get("if_true"),
            if_false=d.get("if_false"),
            on_error=d.get("on_error"),
        )


@dataclass
class WarningCommand(CommandDef):
    """変換できないレガシーActionを表す特別ノード。

    GUI側はこれを警告表示してユーザーの手動修正を促す。
    """

    warning: str = "Unconverted legacy action"
    original_action: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"warning": self.warning, "original_action": self.original_action})
        base["_warning_command"] = True
        return base

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WarningCommand":
        obj = cls(
            id=d.get("id", _new_id()),
            type=d.get("type", "WARNING"),
            params=d.get("params", {}),
            input_keys=d.get("input_keys", []),
            output_keys=d.get("output_keys", []),
            if_true=d.get("if_true"),
            if_false=d.get("if_false"),
            on_error=d.get("on_error"),
            warning=d.get("warning", "Unconverted legacy action"),
            original_action=d.get("original_action"),
        )
        return obj


__all__ = ["CommandDef", "WarningCommand"]
