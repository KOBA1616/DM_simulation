"""
Compatibility wrapper helpers for legacy action/command aliases.

Provides functions to annotate converted Command dicts with legacy aliases
or original action type hints to support short-term compatibility while the
codebase migrates to unified command types.
"""
from typing import Dict, Any


def add_aliases_to_command(cmd: Dict[str, Any], original_action_type: str) -> Dict[str, Any]:
    """
    Annotate the command dict with legacy alias information.

    - Adds `legacy_original_type` if not present.
    - Adds `aliases` list containing the original action type when applicable.

    Returns the modified cmd (in-place and also returned for convenience).
    """
    if not isinstance(cmd, dict):
        return cmd
    if 'legacy_original_type' not in cmd:
        cmd['legacy_original_type'] = original_action_type
    # Preserve explicit reason if present, but also expose aliases
    aliases = set(cmd.get('aliases', []))
    if original_action_type and original_action_type not in aliases:
        aliases.add(original_action_type)
    # If a reason field exists (e.g., DESTROY mapped to TRANSITION), include it
    reason = cmd.get('reason')
    if reason:
        aliases.add(reason)
    if aliases:
        cmd['aliases'] = list(aliases)
    return cmd
