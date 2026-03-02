# -*- coding: utf-8 -*-
"""Legacy ActionConverter compatibility shim.

再発防止: dm_toolkitの Action->Command 変換モジュールは削除済み。
古いテストや統合が ActionConverter を使用する場合の互換シム。
"""

from __future__ import annotations

from typing import Any, Dict

# 再発防止: dm_toolkit.action_to_command は削除済み。ローカル版 map_action で代替。
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
