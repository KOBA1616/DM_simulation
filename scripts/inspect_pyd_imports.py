import os
import sys
from pathlib import Path
try:
    import pefile
except Exception as e:
    print('pefile import error:', e)
    sys.exit(2)

p = Path('native_prototypes') / 'index_to_command' / 'build' / 'Release' / 'index_to_command_native.cp312-win_amd64.pyd'
print('looking for:', p.resolve())
if not p.exists():
    print('pyd not found at expected path')
    sys.exit(1)
pe = pefile.PE(str(p))
imports = [entry.dll.decode('utf-8') for entry in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', [])]
print('imports:')
for d in imports:
    print('  ', d)
