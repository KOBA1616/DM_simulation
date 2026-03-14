"""Compatibility shim: import canonicalize from transforms.normalize_command.

Existing imports of `dm_toolkit.gui.editor.normalize.canonicalize` will continue
to work; prefer importing from `dm_toolkit.gui.editor.transforms.normalize_command`.
"""
from warnings import warn

try:
    from dm_toolkit.gui.editor.transforms.normalize_command import canonicalize  # type: ignore
except Exception:
    # Fallback definition in case import fails in minimal test environments
    def canonicalize(node):
        warn("normalize_command canonicalize shim used; consider importing from transforms.normalize_command", DeprecationWarning)
        if not isinstance(node, dict):
            return {"kind": "UNKNOWN", "payload": node}
        return {"kind": "UNKNOWN", "payload": node, 'uid': node.get('uid')}
