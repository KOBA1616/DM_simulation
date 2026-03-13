# -*- coding: utf-8 -*-
"""Legacy compatibility shim for old action->command conversion.

再発防止: dm_toolkit の旧形式アクション→コマンド変換モジュールは削除済み。
古いテストや統合が利用する際の互換シムを提供します。
"""

from __future__ import annotations

from typing import Any, Dict

# 再発防止: dm_toolkit.action_to_command は削除済み。ローカル版の変換で代替します。
def _map_action_local(action: Any) -> Dict[str, Any]:
    if isinstance(action, dict):
        return action
    if hasattr(action, 'to_dict'):
        try:
            return action.to_dict()
        except Exception:
            pass
    return {'type': str(getattr(action, 'type', 'UNKNOWN'))}


class ActionConverter:
    @staticmethod
    def convert(action: Any) -> Dict[str, Any]:
        return _map_action_local(action)
