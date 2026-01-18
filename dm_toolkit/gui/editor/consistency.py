# -*- coding: utf-8 -*-
"""
Consistency checks for Trigger Scope / Trigger Filter forms.

Detects common duplication or conflicting settings when saving GUI forms
to effect dictionaries used by text generation.
"""
from typing import Dict, List


def validate_trigger_scope_filter(effect: Dict[str, object]) -> List[str]:
    """Validate an effect dict's trigger scope/filter settings.

    Returns a list of human-readable warning messages describing potential
    duplication or conflicts. This is non-fatal and intended for editor-side
    validation or debug output.
    """
    warnings: List[str] = []

    if not isinstance(effect, dict):
        return ["effect is not a dict"]

    scope = str(effect.get("trigger_scope", "NONE") or "NONE").upper()
    f = effect.get("trigger_filter") or {}
    if not isinstance(f, dict):
        f = {}

    # 1) Owner duplication between scope and filter.owner
    owner = str(f.get("owner", "NONE") or "NONE").upper()
    if scope in ("SELF", "PLAYER_SELF") and owner in ("SELF", "PLAYER_SELF"):
        warnings.append("重複: Trigger Scope=自分 と Filter.owner=自分 が重複しています（ownerは未設定推奨）")
    if scope in ("OPPONENT", "PLAYER_OPPONENT") and owner in ("OPPONENT", "PLAYER_OPPONENT"):
        warnings.append("重複: Trigger Scope=相手 と Filter.owner=相手 が重複しています（ownerは未設定推奨）")

    # 2) Cost field conflicts: exact_cost vs min/max/cost_ref
    exact_cost = f.get("exact_cost")
    min_cost = f.get("min_cost")
    max_cost = f.get("max_cost")
    cost_ref = f.get("cost_ref")
    if exact_cost is not None:
        if min_cost not in (None, 0) or max_cost not in (None, 999) or cost_ref:
            warnings.append("競合: ExactCost と Min/Max/CostRef が同時指定されています（ExactCost優先、他は未設定推奨）")

    # 3) Power field conflicts: both dict-linked and numeric ranges simultaneously
    min_power = f.get("min_power")
    max_power = f.get("max_power")
    if isinstance(min_power, dict) and isinstance(max_power, int) and max_power not in (0, 999999):
        warnings.append("警告: 最小パワーが入力連携、最大パワーが数値指定です（片方の形式に統一推奨）")
    if isinstance(max_power, dict) and isinstance(min_power, int) and min_power not in (0,):
        warnings.append("警告: 最大パワーが入力連携、最小パワーが数値指定です（片方の形式に統一推奨）")

    # 4) Zones/type mutual exclusivity sanity (e.g., Shield Zone with Creature type)
    zones = [str(z).upper() for z in (f.get("zones") or [])]
    types = [str(t).upper() for t in (f.get("types") or [])]
    if "SHIELD_ZONE" in zones and ("CREATURE" in types or "SPELL" in types or "ELEMENT" in types):
        warnings.append("不整合: シールドゾーンに対してタイプ（クリーチャー/呪文/エレメント）指定があります（カード指定推奨）")

    # 5) Evolution flag with non-creature type
    is_evo = f.get("is_evolution")
    if is_evo and "CREATURE" not in types:
        warnings.append("不整合: 進化フラグがタイプ=クリーチャー以外で指定されています")

    # 6) Tapped flag with non-battle zone
    is_tapped = f.get("is_tapped")
    if is_tapped is not None and "BATTLE_ZONE" not in zones:
        warnings.append("注意: タップ/アンタップは通常バトルゾーンでのみ意味があります（ゾーン確認）")

    return warnings
