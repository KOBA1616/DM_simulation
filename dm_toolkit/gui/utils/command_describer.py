# -*- coding: utf-8 -*-
from typing import Any
from ..i18n import tr
from .card_helpers import get_card_name_by_instance
import logging

logger = logging.getLogger(__name__)

m: Any = None
try:
    import dm_ai_module as m  # type: ignore
except ImportError:
    pass

def describe_command(cmd: Any, game_state: Any, card_db: Any) -> str:
    """Generate a localized string description for a GameCommand."""
    if not m:
        return "GameCommand（ネイティブモジュール未ロード）"

    # Get command type from class name
    cmd_type = type(cmd).__name__
    
    # Handle each command type based on class name
    if cmd_type == 'TransitionCommand':
        try:
            card_instance_id = getattr(cmd, 'card_instance_id', None)
            owner_id = getattr(cmd, 'owner_id', '?')
            from_zone = getattr(cmd, 'from_zone', '?')
            to_zone = getattr(cmd, 'to_zone', '?')
            name = get_card_name_by_instance(game_state, card_db, card_instance_id)
            return f"[{tr('TRANSITION')}] {name} (P{owner_id}): {tr(from_zone)} -> {tr(to_zone)}"
        except Exception as e:
            return f"[{tr('TRANSITION')}]"
    
    elif cmd_type == 'MutateCommand':
        try:
            target_instance_id = getattr(cmd, 'target_instance_id', None)
            mutation_type = getattr(cmd, 'mutation_type', '?')
            name = get_card_name_by_instance(game_state, card_db, target_instance_id)
            return f"[{tr('MUTATE')}] {name}: {tr(mutation_type)}"
        except Exception:
            return f"[{tr('MUTATE')}]"
    
    elif cmd_type == 'FlowCommand':
        try:
            flow_type = getattr(cmd, 'flow_type', '?')
            return f"[{tr('FLOW')}] {tr(flow_type)}"
        except Exception:
            return f"[{tr('FLOW')}]"
    
    elif cmd_type == 'PlayCardCommand':
        try:
            card_instance_id = getattr(cmd, 'card_instance_id', None)
            name = get_card_name_by_instance(game_state, card_db, card_instance_id) or "Card"
            return f"[{tr('PLAY_CARD')}] {name}"
        except Exception:
            return f"[{tr('PLAY_CARD')}]"
    
    elif cmd_type == 'ManaChargeCommand':
        return f"[{tr('MANA_CHARGE')}]"
    
    elif cmd_type == 'PassCommand':
        return f"[{tr('PASS')}]"
    
    elif cmd_type == 'QueryCommand':
        try:
            query_type = getattr(cmd, 'query_type', '?')
            return f"[{tr('QUERY')}] {tr(query_type)}"
        except Exception:
            return f"[{tr('QUERY')}]"
    
    elif cmd_type == 'ResolveEffectCommand':
        try:
            slot_index = getattr(cmd, 'slot_index', '?')
            return f"[{tr('RESOLVE_EFFECT')}] #{slot_index}"
        except Exception:
            return f"[{tr('RESOLVE_EFFECT')}]"
    
    elif cmd_type == 'DecideCommand':
        return f"[{tr('DECIDE')}]"
    
    return f"{cmd_type}"
