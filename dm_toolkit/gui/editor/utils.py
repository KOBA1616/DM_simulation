# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from typing import Any, Iterable


def safe_load_json(primary_path: str, fallback_paths: Iterable[str] | None = None) -> Any:
    """Try to load JSON from primary_path, otherwise try fallback_paths in order.

    Returns the parsed JSON on success, or None if all attempts fail.
    """
    paths = [primary_path]
    if fallback_paths:
        paths.extend(fallback_paths)

    for p in paths:
        try:
            if not p:
                continue
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            # Swallow errors and try next path
            continue
    return None
# -*- coding: utf-8 -*-

def normalize_action_zone_keys(data):
    """
    Ensures 'source_zone' and 'destination_zone' are present in data if 'from_zone'to_zone' exist.
    Removes 'from_zone' and 'to_zone' to prevent ambiguity and ensure compliance with C++ ActionDef.

    Args:
        data (dict): The action data dictionary to normalize.

    Returns:
        dict: The normalized data (modified in-place).
    """
    if not isinstance(data, dict):
        return data

    # 1. Normalize Source
    if 'source_zone' not in data and 'from_zone' in data:
        data['source_zone'] = data['from_zone']

    # 2. Normalize Destination
    if 'destination_zone' not in data and 'to_zone' in data:
        data['destination_zone'] = data['to_zone']

    # 3. Cleanup Legacy Keys
    if 'from_zone' in data: del data['from_zone']
    if 'to_zone' in data: del data['to_zone']

    return data

def normalize_command_zone_keys(data):
    """
    Ensures 'from_zone' and 'to_zone' are present in data if 'source_zone'destination_zone' exist.
    Removes 'source_zone' and 'destination_zone' to ensure compliance with C++ CommandDef.

    Args:
        data (dict): The command data dictionary to normalize.

    Returns:
        dict: The normalized data (modified in-place).
    """
    if not isinstance(data, dict):
        return data

    # 1. Normalize From
    if 'from_zone' not in data and 'source_zone' in data:
        data['from_zone'] = data['source_zone']

    # 2. Normalize To
    if 'to_zone' not in data and 'destination_zone' in data:
        data['to_zone'] = data['destination_zone']

    # 3. Cleanup Legacy Keys (if they accidentally leaked into Command objects)
    if 'source_zone' in data: del data['source_zone']
    if 'destination_zone' in data: del data['destination_zone']

    return data

# LEGACY_CMDDEF_REFERENCE: This file references 'ActionDef' (legacy). Consider migrating to 'CommandDef'.
