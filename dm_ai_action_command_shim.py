"""
簡易 Action→Command 翻訳 shim。
目的: 移行作業を小刻みに進めるためのヘルパー関数群を提供する。
- 現在は最小限のマッピングのみを実装。段階的に拡張して `dm_ai_module.py` に統合します。
"""
from typing import Any, Dict, Optional
import sys
import os

# Ensure we can import dm_toolkit if it's in the same directory structure
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

try:
    from dm_toolkit.commands import BaseCommand
except ImportError:
    # Fallback if not available
    class BaseCommand:  # type: ignore
        def execute(self, state: Any) -> Optional[Any]:
            raise NotImplementedError()
        def invert(self, state: Any) -> Optional[Any]:
            return None
        def to_dict(self) -> Dict[str, Any]:
            return {"kind": self.__class__.__name__}

class Command(BaseCommand):
    """基底コマンド（shim用）"""
    def __init__(self, kind: str, payload: Dict[str, Any] | None = None):
        self.kind = kind
        self.payload = payload or {}

    def __repr__(self) -> str:
        return f"Command(kind={self.kind!r}, payload={self.payload!r})"

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload}

class MutateCommand(Command):
    def __init__(self, payload: Dict[str, Any] | None = None):
        super().__init__("mutate", payload)

class FlowCommand(Command):
    def __init__(self, payload: Dict[str, Any] | None = None):
        super().__init__("flow", payload)

def translate_action_to_command(action: Any) -> Command:
    """
    単純な翻訳ルール（拡張版）:
    - action.type が文字列/列挙子名として扱えることを期待
    - 既知の type を Mutate/Flow 等へマップして Command を返す

    action は少なくとも以下の属性があると仮定しています:
      - type
      - amount / cost / keyword / reference など（存在する場合）
    """
    atype = getattr(action, "type", None)
    if atype is None:
        return Command("noop", {"reason": "no type"})

    # 文字列比較を基本にする（ENUM の `.name` にも対応）
    name = atype.name if hasattr(atype, "name") else str(atype)
    name = name.replace("'", "").replace("\"", "") # Cleanup

    # --- High Priority ---
    if name == "BLOCK":
        blocker = getattr(action, 'blocker_id', getattr(action, 'source_instance_id', None))
        attacker = getattr(action, 'attacker_id', getattr(action, 'target_instance_id', None))
        return FlowCommand({"block": True, "blocker": blocker, "attacker": attacker})

    if name == "BREAK_SHIELD":
        # Usually internal, but maps to a Mutate or Flow
        creature = getattr(action, 'creature_id', getattr(action, 'source_instance_id', None))
        shield = getattr(action, 'shield_id', getattr(action, 'target_instance_id', None))
        return MutateCommand({"break_shield": True, "creature": creature, "shield": shield})

    if name == "RESOLVE_BATTLE":
        winner = getattr(action, 'winner_id', None)
        return FlowCommand({"resolve_battle": True, "winner": winner})

    if name == "RESOLVE_EFFECT":
        effect_id = getattr(action, 'effect_id', None)
        return FlowCommand({"resolve_effect": True, "effect_id": effect_id})

    if name == "USE_SHIELD_TRIGGER":
        card = getattr(action, 'card_id', getattr(action, 'source_instance_id', None))
        return FlowCommand({"use_shield_trigger": True, "card": card})

    # --- Medium Priority ---
    if name == "DESTROY":
        target = getattr(action, 'target_id', getattr(action, 'instance_id', None))
        return MutateCommand({"destroy": target})

    if name == "DISCARD":
        target = getattr(action, 'target_id', getattr(action, 'card_id', None))
        return MutateCommand({"discard": target})

    if name in ("TAP", "UNTAP"):
        target = getattr(action, 'target_id', getattr(action, 'instance_id', None))
        return MutateCommand({name.lower(): target})

    # --- Existing & Low Priority ---

    # Basic payment / mana
    if name in ("ADD_MANA", "MANA_CHARGE", "CHARGE_MANA", "SEND_TO_MANA"):
        amount = getattr(action, "amount", getattr(action, "value", 1))
        # Handle if it's a specific card move
        card = getattr(action, 'card_id', None)
        try:
            amount = int(amount)
        except Exception:
            amount = 1
        return MutateCommand({"add_mana": amount, "card_id": card})

    # Draw / search
    if name in ("DRAW_CARD", "SEARCH_DECK_BOTTOM", "SEARCH_DECK"):
        count = getattr(action, "count", getattr(action, 'value1', 1))
        try:
            count = int(count)
        except Exception:
            count = 1
        return MutateCommand({"draw": count})

    # Play / declare
    if name in ("PLAY_CARD", "DECLARE_PLAY", "PLAY_CARD_INTERNAL", "RESOLVE_PLAY", "CAST_SPELL", "PUT_CREATURE"):
        cid = getattr(action, 'card_id', getattr(action, 'card', None))
        source = getattr(action, 'source_zone', getattr(action, 'zone', None))
        cost = getattr(action, 'cost', getattr(action, 'amount', None))
        return FlowCommand({"play": True, "card_id": cid, "source": source, "cost": cost})

    # Pay cost
    if name in ("PAY_COST", "PAY_MANA"):
        cost = getattr(action, "cost", getattr(action, "amount", None))
        return MutateCommand({"pay_cost": cost})

    # Grant / keyword
    if name in ("GRANT_KEYWORD", "APPLY_MODIFIER", "MODIFY_POWER"):
        kw = getattr(action, "keyword", getattr(action, "ability", None))
        val = getattr(action, "value", None)
        return MutateCommand({"grant_keyword": kw, "value": val, "type": name})

    if name == "COST_REFERENCE":
        ref = getattr(action, "reference", None)
        return MutateCommand({"cost_reference": ref})

    # Selection / reaction / decision
    if name in ("DECLARE_REACTION", "CHOICE", "DECIDE", "SELECT_OPTION", "SELECT_NUMBER", "USE_ABILITY"):
        val = getattr(action, "value", None)
        return FlowCommand({"action_name": name, "value": val})

    # Attack
    if name in ("ATTACK_PLAYER", "ATTACK_CREATURE"):
        attacker = getattr(action, 'attacker_id', getattr(action, 'source_instance_id', getattr(action, 'instance_id', None)))
        target = getattr(action, 'target_instance_id', getattr(action, 'target_id', getattr(action, 'defender_id', None)))
        return FlowCommand({"attack": True, "attacker": attacker, "target": target})

    if name in ("RETURN_TO_HAND", "MOVE_TO_HAND"):
        cid = getattr(action, 'card_id', getattr(action, 'target_id', None))
        return MutateCommand({"return_to_hand": cid})

    if name in ("SELECT_TARGET", "SELECT_FROM_BUFFER", "LOOK_TO_BUFFER"):
        slot = getattr(action, 'slot_index', getattr(action, 'slot', 0))
        tid = getattr(action, 'target_instance_id', getattr(action, 'target', None))
        return FlowCommand({"select_target": True, "slot": slot, "target": tid})

    if name == "MOVE_CARD":
        cid = getattr(action, 'card_id', None)
        dest = getattr(action, 'destination', None)
        return MutateCommand({"move_card": cid, "destination": dest})

    if name == "PASS":
        return FlowCommand({"pass": True})

    # Fallback for others (Low Priority)
    # Just wrap the name and available attributes
    return Command("generic", {"action_type": name})

# 簡易デモ用 main（手動で試すときだけ）
if __name__ == "__main__":
    class A: pass
    a = A()
    a.type = "ADD_MANA"
    a.amount = 3
    print(translate_action_to_command(a))
