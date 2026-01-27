"""Proxy module for `dm_ai_module`.

The canonical loader/stub lives at repository root as `dm_ai_module.py`.
This package module exists so internal code can consistently import via
`from dm_toolkit import dm_ai_module`.
"""

import importlib

# Import the top-level shim module and copy public names into this package module's
# namespace. Using importlib avoids accidental self-referencing when package and
# module share the same name during import-time initialization.
_root = importlib.import_module('dm_ai_module')
for _name in dir(_root):
	if _name.startswith('__'):
		continue
	try:
		globals()[_name] = getattr(_root, _name)
	except Exception:
		pass
