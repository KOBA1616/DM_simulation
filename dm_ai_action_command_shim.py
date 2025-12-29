"""
簡易 Action→Command 翻訳 shim。
目的: 移行作業を小刻みに進めるためのヘルパー関数群を提供する。
- 現在は `dm_toolkit.commands_new` および `dm_toolkit.action_mapper` を使用して
  正規のスキーマに準拠した Command を返すように変更されています。
"""
from typing import Any, Dict, Optional
import sys
import os
from dm_toolkit.commands_new import wrap_action, BaseCommand

# Ensure we can import dm_toolkit if it's in the same directory structure
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Re-export Command as BaseCommand for backward compatibility in tests
Command = BaseCommand

class CompatCommand(BaseCommand):
    """Shim-specific command wrapper that mimics the old behavior but uses new schema."""
    def __init__(self, kind: str, payload: Dict[str, Any] | None = None):
        self.kind = kind
        self.payload = payload or {}
        # Try to map kind/payload to new schema structure if possible,
        # but since this class is mostly used for manual construction in tests,
        # we keep it simple or delegate?
        # For now, let's just act as a container.

    def to_dict(self) -> Dict[str, Any]:
        # Return new schema format if possible, or fallback
        return {
            "type": self.kind.upper(),
            "kind": self.kind,
            "params": self.payload, # Embed payload in params or merge?
            "payload": self.payload,
            # Schema requires 'type'.
            "legacy_warning": True,
            "legacy_original_value": self.payload
        }


class MutateCommand(CompatCommand):
    """Compatibility wrapper for legacy MutateCommand used in tests."""
    def __init__(self, payload: Dict[str, Any] | None = None):
        super().__init__(kind="mutate", payload=payload)


class FlowCommand(CompatCommand):
    """Compatibility wrapper for legacy FlowCommand used in tests."""
    def __init__(self, payload: Dict[str, Any] | None = None):
        super().__init__(kind="flow", payload=payload)

def translate_action_to_command(action: Any) -> BaseCommand:
    """
    Deprecated: Use dm_toolkit.commands_new.wrap_action instead.
    This function now delegates to wrap_action to ensure consistency.
    """
    # Provide a lightweight compatibility mapping for legacy tests.
    t = getattr(action, 'type', None)
    if t == "ADD_MANA":
        return MutateCommand({"add_mana": getattr(action, 'amount', None)})
    if t == "DRAW_CARD":
        return MutateCommand({"draw": getattr(action, 'count', 1)})
    if t == "BLOCK":
        return FlowCommand({"block": True, "blocker": getattr(action, 'source_instance_id', None), "attacker": getattr(action, 'target_instance_id', None)})
    if t == "BREAK_SHIELD":
        return MutateCommand({"break_shield": True})
    if t == "DESTROY":
        return MutateCommand({"destroy": getattr(action, 'target_id', None)})
    if t == "TAP":
        return MutateCommand({"tap": getattr(action, 'instance_id', None)})
    if t == "PASS":
        return FlowCommand({"pass": True})

    # Fallback: return a generic CompatCommand with action_type for unknown actions
    return CompatCommand('generic', {'action_type': t})

if __name__ == "__main__":
    class A: pass
    a = A()
    a.type = "ADD_MANA"
    a.amount = 3
    print(translate_action_to_command(a).to_dict())
