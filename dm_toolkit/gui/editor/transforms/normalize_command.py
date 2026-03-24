# -*- coding: utf-8 -*-
"""Normalization helpers moved to transforms/ for clearer responsibility.

Provides `canonicalize(node)` which returns a canonical internal
representation (CIR) for command-like nodes. Intended to be a pure,
non-destructive view used by editor logic.
"""
from typing import Dict, Any, List


def _normalize_options_list(options) -> List[List[Dict[str, Any]]]:
    out = []
    if not options:
        return out
    for opt in options:
        if isinstance(opt, list):
            out.append([o.copy() if isinstance(o, dict) else o for o in opt])
        else:
            out.append([opt.copy() if isinstance(opt, dict) else opt])
    return out


def canonicalize(node: Dict[str, Any]) -> Dict[str, Any]:
    """Return a canonical internal representation (CIR) for command-like nodes.

    CIR fields: kind, type, options, branches, payload, uid
    """
    if not isinstance(node, dict):
        return {"kind": "UNKNOWN", "payload": node}

    payload = node.copy()
    uid = payload.get('uid')

    # Heuristic: branches / options present -> treat as COMMAND
    if 'if_true' in payload or 'if_false' in payload or ('options' in payload and not payload.get('type')):
        branches = {
            'if_true': [c.copy() for c in payload.get('if_true', [])] if payload.get('if_true') else [],
            'if_false': [c.copy() for c in payload.get('if_false', [])] if payload.get('if_false') else []
        }
        options = _normalize_options_list(payload.get('options', []))
        return {
            'kind': 'COMMAND',
            'type': payload.get('type'),
            'branches': branches,
            'options': options,
            'payload': payload,
            'uid': uid
        }

    if 'type' in payload:
        options = _normalize_options_list(payload.get('options', []))
        return {
            'kind': 'COMMAND',
            'type': payload.get('type'),
            'options': options,
            'payload': payload,
            'uid': uid
        }

    return {"kind": "UNKNOWN", "payload": payload, 'uid': uid}
