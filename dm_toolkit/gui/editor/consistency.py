# -*- coding: utf-8 -*-
"""
Consistency checks for Trigger Scope / Trigger Filter forms, and command logic.

Detects common duplication or conflicting settings when saving GUI forms
to effect dictionaries used by text generation.
"""
from typing import Any, Dict, List, Union


def format_integrity_warnings(warnings: List[str]) -> str:
    """整合性警告メッセージを UI/ログ共通フォーマットへ整形する。"""
    if not warnings:
        return ""
    # 再発防止: 警告文言の組み立てを 1 箇所に集約し、表示揺れを防ぐ。
    lines = ["整合性警告:"]
    lines.extend(f"- {msg}" for msg in warnings)
    return "\n".join(lines)


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


# ──────────────────────────────────────────────
# Command-level consistency checks
# ──────────────────────────────────────────────

_IF_LIKE_TYPES = {"IF", "IF_ELSE", "CONDITIONAL"}
_MULTI_OPTION_TYPES = {"SELECT_OPTION", "CHOICE", "PLAYER_CHOICE"}


def _normalize_zone_token(zone: Any) -> str:
    """Normalize zone tokens so validation can compare legacy and current keys safely."""
    if zone is None:
        return ""
    z = str(zone).strip().upper()
    # 再発防止: UI/legacy の揺れ（fromZone/source_zone 等）を同一判定できるよう正規化する。
    aliases = {
        "HAND": "HAND",
        "MANA": "MANA_ZONE",
        "MANA_ZONE": "MANA_ZONE",
        "BATTLE": "BATTLE_ZONE",
        "BATTLE_ZONE": "BATTLE_ZONE",
        "GRAVE": "GRAVEYARD",
        "GRAVEYARD": "GRAVEYARD",
        "DECK": "DECK",
        "DECK_BOTTOM": "DECK_BOTTOM",
        "SHIELD": "SHIELD_ZONE",
        "SHIELD_ZONE": "SHIELD_ZONE",
        "NONE": "NONE",
    }
    return aliases.get(z, z)


def _extract_filter_zones(params: Dict[str, Any]) -> List[str]:
    """Extract normalized filter zones from either target_filter or filter payload."""
    f = params.get("target_filter") or params.get("filter") or {}
    if not isinstance(f, dict):
        return []
    zones = f.get("zones") or []
    out: List[str] = []
    for z in zones:
        nz = _normalize_zone_token(z)
        if nz and nz not in ("NONE",):
            out.append(nz)
    return out


def _extract_command_fields(cmd: Any):
    """CommandModel または dict から (cmd_type, params, options, if_true, if_false) を返す.

    NOTE: CommandModel は params に str_param 等を格納するが、
    直接属性アクセスはできないため params dict 経由で参照する。
    """
    if hasattr(cmd, "type") and hasattr(cmd, "params"):
        # CommandModel インスタンス
        return (
            str(cmd.type).upper(),
            cmd.params if isinstance(cmd.params, dict) else {},
            list(cmd.options) if cmd.options else [],
            list(cmd.if_true) if cmd.if_true else [],
            list(cmd.if_false) if cmd.if_false else [],
        )
    if isinstance(cmd, dict):
        _KNOWN = {"uid", "type", "if_true", "if_false", "options", "input_var", "output_var"}
        params: Dict[str, Any] = {k: v for k, v in cmd.items() if k not in _KNOWN}
        return (
            str(cmd.get("type", "")).upper(),
            params,
            cmd.get("options", []),
            cmd.get("if_true", []),
            cmd.get("if_false", []),
        )
    return (None, {}, [], [], [])


def validate_command_list(
    commands: List[Any],
    _path: str = "root",
) -> List[str]:
    """コマンドリスト（CommandModel または dict のリスト）の整合性チェック.

    戻り値: 人間が読める警告メッセージリスト（空 = 問題なし）

    チェック内容:
    - IF / IF_ELSE に条件(target_filter)が未設定
    - SELECT_OPTION / CHOICE にブランチが 0 個
    - QUERY に str_param (query mode) が未設定
    - QUERY(SELECT_OPTION) で選択対象フィルタ/枚数が未設定（新仕様）
    - QUERY(SELECT_OPTION) の旧形式 str_val とブランチ数の不一致（レガシー互換）
    - TRANSITION で from_zone と target_filter.zones が矛盾
    """
    warnings: List[str] = []

    for i, cmd in enumerate(commands):
        cmd_type, params, options, if_true, if_false = _extract_command_fields(cmd)
        if cmd_type is None:
            continue

        loc = f"{_path}[{i}]({cmd_type})"

        # 1) IF / IF_ELSE に条件未設定
        if cmd_type in _IF_LIKE_TYPES:
            tf = params.get("target_filter")
            if not tf:
                warnings.append(
                    f"未設定: {loc} に条件(target_filter)が設定されていません"
                )

        # 2) SELECT_OPTION / CHOICE にブランチが 0 個
        if cmd_type in _MULTI_OPTION_TYPES:
            if not options:
                warnings.append(
                    f"未設定: {loc} に選択肢ブランチが 1 つもありません"
                )

        # 3) QUERY チェック
        if cmd_type == "QUERY":
            mode = (params.get("str_param") or "").strip()
            if not mode:
                warnings.append(
                    f"未設定: {loc} に Query Mode (str_param) が設定されていません"
                )
            elif mode == "SELECT_OPTION":
                tf = params.get("target_filter")
                amount = params.get("amount")
                input_key = params.get("input_value_key") or params.get("input_var")

                # 新仕様: フィルタ対象カードを count 指定で選択する。
                # amount 直指定または input_value_key 参照のどちらかが必要。
                if not tf:
                    warnings.append(
                        f"未設定: {loc} SELECT_OPTION に対象フィルタ (target_filter) が設定されていません"
                    )
                amt_val = 0
                try:
                    amt_val = int(amount) if amount is not None else 0
                except (TypeError, ValueError):
                    amt_val = 0

                if not input_key and amt_val <= 0:
                    warnings.append(
                        f"未設定: {loc} SELECT_OPTION の選択数 (amount または input_value_key) が設定されていません"
                    )

                # 旧仕様互換: str_val がある場合のみブランチ整合をチェック。
                sv = (params.get("str_val") or "").strip()
                if sv:
                    label_lines = [l for l in sv.split("\n") if l.strip()]
                    branch_count = len(options)
                    if label_lines and branch_count and len(label_lines) != branch_count:
                        warnings.append(
                            f"不一致: {loc} 旧形式選択肢テキスト数 ({len(label_lines)}) と"
                            f" ブランチ数 ({branch_count}) が一致しません"
                        )

        # 4) MOVE 系チェック: 移動元ゾーンとフィルターゾーンの矛盾
        if cmd_type == "TRANSITION":
            from_zone = _normalize_zone_token(
                params.get("from_zone") or params.get("fromZone") or params.get("source_zone")
            )
            filter_zones = _extract_filter_zones(params)

            # 再発防止: from_zone が固定されている場合、filter.zones と矛盾する設定を保存前に検知する。
            if from_zone and from_zone not in ("NONE",) and filter_zones and from_zone not in filter_zones:
                warnings.append(
                    f"競合: {loc} Source Zone={from_zone} と Filter.zones={filter_zones} が一致していません"
                )

        # 再帰チェック
        if if_true:
            warnings.extend(validate_command_list(if_true, f"{loc}.if_true"))
        if if_false:
            warnings.extend(validate_command_list(if_false, f"{loc}.if_false"))
        for j, opt_branch in enumerate(options):
            if isinstance(opt_branch, list):
                warnings.extend(
                    validate_command_list(opt_branch, f"{loc}.options[{j}]")
                )

    return warnings
