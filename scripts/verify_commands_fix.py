
import sys
import os

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from dm_toolkit import commands
    print("Successfully imported dm_toolkit.commands")
except ImportError as e:
    print(f"Failed to import dm_toolkit.commands: {e}")
    sys.exit(1)

if not hasattr(commands, 'generate_legal_commands'):
    print("commands.generate_legal_commands is missing")
    sys.exit(1)

if not hasattr(commands, 'ICommand'):
    print("commands.ICommand is missing")
    sys.exit(1)

print("Verification passed.")
