# -*- coding: utf-8 -*-
"""Lightweight legacy->command conversion helpers.

This module provides conservative converters used by migration/testing paths.
It is intentionally small and deterministic so it can be safely exercised
in unit tests before wider integration.
"""
from typing import Any, Dict


def convert_legacy_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a legacy action payload to a normalized command dict.

        Rules (conservative):
        - If payload contains explicit `type` or `action` key (case-insensitive), use that
      as the command `type`.
    - Move common positional keys (e.g., `card_id`, `amount`) under `params`.
    - Preserve original payload under `legacy_action` for traceability.
    - Never remove keys from the original payload passed in (work on a copy).
    """
    if not isinstance(payload, dict):
        return {"type": "NONE", "legacy_action": payload}

    src = dict(payload)

    # Determine type
    cmd_type = None
    if 'type' in src and isinstance(src['type'], str):
        cmd_type = src['type']
    else:
        # legacy keys (case-insensitive)
        for k, v in src.items():
            if isinstance(k, str) and k.lower() == 'action' and isinstance(v, str):
                cmd_type = v
                break

    if not cmd_type:
        cmd_type = 'LEGACY_CMD'

    # Extract params heuristically
    params = {}
    for k in ('card_id', 'amount', 'count', 'zone', 'target'):
        if k in src:
            params[k] = src[k]

    # If payload already has a params-like dict under 'params', merge it
    if 'params' in src and isinstance(src['params'], dict):
        merged = dict(src['params'])
        merged.update(params)
        params = merged

    out = {"type": cmd_type}
    if params:
        out['params'] = params

    # Keep legacy copy for auditing
    out['legacy_action'] = src

    return out
