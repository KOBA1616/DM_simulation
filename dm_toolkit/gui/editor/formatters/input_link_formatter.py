from typing import Any, Dict, List
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources

class InputLinkFormatter:
    """Formatter for resolving and generating text related to input variable linking."""

    @classmethod
    def format_input_link_text(cls, command: Dict[str, Any], atype: str) -> str:
        """Centralized generation of 'その数' or 'その同じ数' phrase based on input linked tokens."""
        linked_text = cls.resolve_linked_value_text(command)
        if not linked_text:
            return ""

        up_to_flag = bool(command.get('up_to', False))

        # When an input value defines the count of targets to affect:
        if atype in ("DESTROY", "TAP", "UNTAP", "RETURN_TO_HAND", "SEND_TO_MANA"):
            if up_to_flag:
                if atype == "DESTROY":
                    return "その同じ数だけまで選び、"
                return "その同じ数だけまで選び、"
            else:
                if atype == "DESTROY":
                    return "その同じ数だけ"
                return "その同じ数だけ選び、"

        return "その数"

    @classmethod
    def format_input_usage_label(cls, usage: Any) -> str:
        """Return a short label indicating how an input value is used."""
        if usage is None:
            return ""
        norm = str(usage).upper()
        # Suppress label for MAX_COST to avoid redundant parenthetical hints
        if norm == "MAX_COST":
            return ""
        if norm in CardTextResources.INPUT_USAGE_LABELS:
            return CardTextResources.INPUT_USAGE_LABELS[norm]
        # Fallback to raw string for custom labels
        return tr(str(usage)) if str(usage) else ""

    @classmethod
    def format_input_source_label(cls, action: Dict[str, Any]) -> str:
        """Resolve a human-readable source label for input-linked commands."""
        input_key = str(action.get("input_value_key") or "")
        if not input_key:
            return ""

        saved = str(action.get("_input_value_label") or "").strip()
        if saved:
            # 再発防止: UI補足の括弧書きはカード本文に不要なので除去する。
            if "(" in saved:
                saved = saved.split("(", 1)[0].strip()
            if "（" in saved:
                saved = saved.split("（", 1)[0].strip()
            return saved
        if input_key == "EVENT_SOURCE":
            return "イベント発生源 (汎用)"
        return input_key

    @classmethod
    def normalize_linked_count_label(cls, label: str) -> str:
        """Normalize query-derived count labels into natural Japanese wording."""
        text = str(label or "").strip()
        if not text:
            return "その数"
        text = text.replace("カード枚数", "枚数")
        text = text.replace("カードの枚数", "枚数")
        text = text.replace("カード枚", "枚数")
        if text.endswith("枚"):
            return text + "数"
        if text.endswith("体"):
            return text + "数"
        return text

    @classmethod
    def format_linked_count_token(cls, action: Dict[str, Any], fallback: str) -> str:
        """Return count token for linked-input text. Prefer semantic labels over generic wording."""
        label = cls.format_input_source_label(action)
        if not label:
            return fallback
        normalized = label.strip()
        if not normalized:
            return fallback
        if normalized.startswith("Step ") or normalized in ("クエリ結果",):
            return fallback
        if normalized in ("引いた枚数", "捨てた枚数", "選択した数"):
            return normalized
        return normalized

    @classmethod
    def infer_output_value_label(cls, command: Dict[str, Any]) -> str:
        """Infer a human-readable output label from command semantics."""
        ctype = str(command.get("type") or command.get("name") or "")
        if ctype == "QUERY":
            mode = str(command.get("str_param") or command.get("query_mode") or command.get("str_val") or "")
            if mode in CardTextResources.STAT_KEY_MAP:
                stat_name, stat_unit = CardTextResources.STAT_KEY_MAP[mode]
                if stat_unit:
                    return f"{stat_name}{stat_unit}数"
                return stat_name
            if mode:
                return tr(mode)
        if ctype == "DRAW_CARD":
            return "引いた枚数"
        if ctype == "DISCARD":
            return "捨てた枚数"
        if ctype in ("DECLARE_NUMBER", "SELECT_NUMBER"):
            return "選択した数"
        return ""

    @classmethod
    def format_input_link_context_suffix(cls, action: Dict[str, Any]) -> str:
        """Format input-link metadata used in card preview text."""
        source_label = cls.format_input_source_label(action)
        usage_raw = action.get("input_value_usage") or action.get("input_usage")
        usage_label = cls.format_input_usage_label(usage_raw) if usage_raw else ""

        parts: List[str] = []
        if source_label:
            parts.append(f"入力元: {source_label}")
        if usage_raw and usage_label:
            parts.append(f"入力用途: {usage_label}")

        if not parts:
            return ""
        return f"（{' / '.join(parts)}）"

    @classmethod
    def resolve_linked_value_text(cls, command: Dict[str, Any], default: str = "") -> str:
        """Resolve the linked value text for a given command.

        Returns a string like 'その数' or 'そのコストと同じ' if the command has an input link,
        otherwise returns the default string provided.
        """
        input_key = str(command.get("input_value_key") or command.get("input_link") or "")
        if not input_key:
            return default

        # Try to resolve a specific label if it was saved during UI editing
        input_label = command.get("_input_value_label", "")
        if not input_label:
             input_label = cls.format_input_source_label(command)

        usage = str(command.get("input_usage") or command.get("input_value_usage") or "").upper()

        if usage in ("COST", "MAX_COST"):
             return "そのコスト以下"
        elif usage in ("POWER", "MAX_POWER"):
             return "そのパワー以下"

        # fallback for count/amount
        if input_label:
            return "その数"

        return default
