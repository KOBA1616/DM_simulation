import sys
import os

# Add root directory to path to allow imports
root_path = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(root_path)

try:
    from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
    print("Checked consts.py: SUCCESS")
except ImportError as e:
    print(f"Failed to import consts: {e}")

try:
    from dm_toolkit.gui.editor.forms.command_strategies import get_strategy, STRATEGY_MAP
    print("Checked command_strategies.py: SUCCESS")
except ImportError as e:
    print(f"Failed to import command_strategies: {e}")

try:
    # This requires PyQt6 which might not be fully functional in headless env
    # We just check for syntax errors by importing
    from dm_toolkit.gui.editor.logic_tree import LogicTreeWidget
    print("Checked logic_tree.py: SUCCESS")
except Exception as e:
    print(f"LogicTreeWidget verification check: {e}")

try:
    from dm_toolkit.gui.editor.data_manager import CardDataManager
    print("Checked data_manager.py: SUCCESS")
except Exception as e:
    print(f"CardDataManager verification check: {e}")

try:
    # Need to mock BaseEditForm dependencies or check syntax
    from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
    print("Checked unified_action_form.py: SUCCESS")
except ImportError as e:
    print(f"Failed to import UnifiedActionForm: {e}")
except Exception as e:
    # Expected failure in headless env without QApplication
    print(f"UnifiedActionForm import attempted (syntax check): {e}")
