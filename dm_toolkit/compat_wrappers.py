"""
Compatibility Wrapper Helpers for Legacy Action/Command Aliases

This module centralizes backward compatibility logic as mandated by AGENTS.md Policy Section 2:
"Minimize Dispersion: Logic for backward compatibility and command post-processing should
be centralized."

Key Functions:
- add_aliases_to_command: Annotates commands with legacy type information for compatibility
  with older test code and execution paths that expect specific action type strings.
- is_legacy_command: Detects if a command was converted from a legacy Action
- normalize_legacy_fields: Post-processes commands to ensure backward compatibility

Usage Pattern:
    After converting an action to command via action_to_command.map_action,
    the command is automatically annotated with:
    - legacy_original_type: Original action type before mapping
    - aliases: List of equivalent legacy type names for backward compatibility
    
    This allows execution engines and tests to recognize commands by their legacy
    names without modifying the core command structure.

Goal: Avoid spreading "if legacy_mode:" checks throughout the codebase.
"""
from typing import Dict, Any, Optional, cast


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


def is_legacy_command(cmd: Dict[str, Any]) -> bool:
    """
    Detect if a command was converted from a legacy Action dictionary.
    
    Args:
        cmd: Command dictionary to check
        
    Returns:
        True if command has legacy markers, False otherwise
    """
    if not isinstance(cmd, dict):
        return False
    return 'legacy_original_type' in cmd or 'aliases' in cmd or cmd.get('legacy_warning', False)


def normalize_legacy_fields(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process a command to ensure backward compatibility with legacy code paths.
    
    This function centralizes "if legacy_mode" style checks that were previously
    scattered throughout the codebase. It ensures commands have both canonical
    and legacy field names where needed.
    
    Normalization Rules:
    - Ensure both 'str_param' and 'str_val' exist if either is present
    - Ensure both 'amount' and 'value1' exist if either is present
    - Preserve 'value2' if present in either form
    
    Args:
        cmd: Command dictionary to normalize (modified in-place)
        
    Returns:
        Normalized command dictionary
    """
    if not isinstance(cmd, dict):
        return cmd
        
    # String parameter bidirectional compatibility
    if 'str_param' in cmd and 'str_val' not in cmd:
        cmd['str_val'] = cmd['str_param']
    elif 'str_val' in cmd and 'str_param' not in cmd:
        cmd['str_param'] = cmd['str_val']
    
    # Numeric parameter bidirectional compatibility
    if 'amount' in cmd and 'value1' not in cmd:
        cmd['value1'] = cmd['amount']
    elif 'value1' in cmd and 'amount' not in cmd:
        cmd['amount'] = cmd['value1']
    
    return cmd


def get_effective_command_type(cmd: Dict[str, Any]) -> Optional[str]:
    """
    Get the effective command type, considering both canonical and legacy types.
    
    For commands converted from legacy Actions, this returns the original
    action type if the current type is NONE or invalid, providing fallback
    for compatibility.
    
    Args:
        cmd: Command dictionary
        
    Returns:
        Effective command type string, or None if indeterminate
    """
    if not isinstance(cmd, dict):
        return None
    
    current_type = cast(str, cmd.get('type', 'NONE'))
    
    # If current type is valid, use it
    if current_type and current_type != 'NONE':
        # Unless it's marked as a legacy warning (invalid conversion)
        if cmd.get('legacy_warning', False):
            # legacy_original_type may be Any; cast to Optional[str] for mypy
            return cast(Optional[str], cmd.get('legacy_original_type')) or current_type
        return current_type
    
    # Fallback to legacy original type (ensure str | None)
    return cast(Optional[str], cmd.get('legacy_original_type')) or 'NONE'

