"""
簡易 Action→Command 翻訳 shim。
目的: 移行作業を小刻みに進めるためのヘルパー関数群を提供する。
- 現在は最小限のマッピングのみを実装。段階的に拡張して `dm_ai_module.py` に統合します。
"""
from typing import Any, Dict

class Command:
    """基底コマンド（最小実装）"""
    def __init__(self, kind: str, payload: Dict[str, Any] | None = None):
        self.kind = kind
        self.payload = payload or {}

    def __repr__(self) -> str:
        return f"Command(kind={self.kind!r}, payload={self.payload!r})"

class MutateCommand(Command):
    def __init__(self, payload: Dict[str, Any] | None = None):
        super().__init__("mutate", payload)

class FlowCommand(Command):
    def __init__(self, payload: Dict[str, Any] | None = None):
        super().__init__("flow", payload)


def translate_action_to_command(action: Any) -> Command:
    """
    単純な翻訳ルール（拡張予定）:
    - action.type が文字列/列挙子名として扱えることを期待
    - 既知の type を Mutate/Flow 等へマップして Command を返す
    - 未知の action は `Command('noop')` を返す（実装者がログで検出して拡張する）

    action は少なくとも以下の属性があると仮定しています:
      - type
      - amount / cost / keyword / reference など（存在する場合）
    """
    atype = getattr(action, "type", None)
    if atype is None:
        return Command("noop", {"reason": "no type"})

    # 文字列比較を基本にする（ENUM の `.name` にも対応）
    name = atype.name if hasattr(atype, "name") else str(atype)

    if name in ("ADD_MANA", "MANA_CHARGE", "CHARGE_MANA"):
        amount = getattr(action, "amount", getattr(action, "value", 1))
        try:
            amount = int(amount)
        except Exception:
            amount = 1
        return MutateCommand({"add_mana": amount})

    if name in ("DRAW_CARD", "SEARCH_DECK_BOTTOM"):
        count = getattr(action, "count", 1)
        try:
            count = int(count)
        except Exception:
            count = 1
        return MutateCommand({"draw": count})

    if name in ("PLAY_CARD", "DECLARE_PLAY"):
        # include card id, source zone and cost when available
        cid = getattr(action, 'card_id', getattr(action, 'card', None))
        source = getattr(action, 'source_zone', getattr(action, 'zone', None))
        cost = getattr(action, 'cost', getattr(action, 'pay', None))
        return FlowCommand({"play": True, "card_id": cid, "source": source, "cost": cost})

    if name == "PAY_COST" or name == "PAY_MANA":
        cost = getattr(action, "cost", getattr(action, "amount", None))
        return MutateCommand({"pay_cost": cost})

    if name == "GRANT_KEYWORD":
        kw = getattr(action, "keyword", getattr(action, "ability", None))
        return MutateCommand({"grant_keyword": kw})

    if name == "COST_REFERENCE":
        ref = getattr(action, "reference", None)
        return MutateCommand({"cost_reference": ref})

    # デフォルト: flow/decide にマップする試み
    if name in ("DECLARE_REACTION", "CHOICE", "DECIDE"):
        return FlowCommand({"action_name": name})

    if name in ("ATTACK_PLAYER", "ATTACK_CREATURE"):
        attacker = getattr(action, 'attacker_id', getattr(action, 'instance_id', None))
        target = getattr(action, 'target_id', getattr(action, 'defender_id', None))
        return FlowCommand({"attack": True, "attacker": attacker, "target": target})

    if name in ("RETURN_TO_HAND", "MOVE_TO_HAND"):
        cid = getattr(action, 'card_id', None)
        return MutateCommand({"return_to_hand": cid})

    # 未知のアクションは noop コマンドとして返し、後で拡張する
    return Command("noop", {"action_type": name})


# 簡易デモ用 main（手動で試すときだけ）
if __name__ == "__main__":
    class A: pass

    a = A()
    a.type = "ADD_MANA"
    a.amount = 3
    print(translate_action_to_command(a))
