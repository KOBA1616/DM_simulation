from typing import Dict, Any, Tuple
from dm_toolkit.consts import TargetScope
from dm_toolkit.gui.editor.text_resources import CardTextResources

class TargetScopeResolver:
    """
    Utility class for resolving TargetScope to standardized Japanese prefixes or nouns.
    Eliminates hardcoded strings like "自分の", "自分", "相手の", "相手" scattered across the codebase.
    """

    @classmethod
    def resolve_prefix(cls, scope: str, default: str = "") -> str:
        """
        Resolves the TargetScope to a Japanese prefix (e.g. "自分の", "相手の").
        """
        if not scope or scope == TargetScope.ALL or scope == "NONE":
            return default

        normalized = TargetScope.normalize(scope)
        text = CardTextResources.get_scope_text(normalized)
        return text if text else default

    @classmethod
    def resolve_noun(cls, scope: str, default: str = "") -> str:
        """
        Resolves the TargetScope to a Japanese noun representing the player (e.g. "自分", "相手", "すべてのプレイヤー").
        """
        if not scope:
            return default

        normalized = TargetScope.normalize(scope)

        if normalized == TargetScope.SELF:
            return "自分"
        elif normalized == TargetScope.OPPONENT:
            return "相手"
        elif normalized == TargetScope.ALL or normalized == "ALL_PLAYERS":
            return "すべてのプレイヤー"
        elif normalized == "NONE":
             return default

        # Try to resolve via prefix if it's an unknown scope type
        prefix = cls.resolve_prefix(normalized)
        if prefix and prefix.endswith("の"):
            return prefix[:-1]

        return default

    @classmethod
    def resolve_action_scope(cls, action: Dict[str, Any]) -> str:
        """
        Resolves the scope from an action dictionary, preferring 'target_group' over 'scope'.
        Returns the normalized TargetScope.
        """
        scope = action.get("target_group") or action.get("scope", "NONE")
        return TargetScope.normalize(scope)
