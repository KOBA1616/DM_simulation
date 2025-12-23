import sys
import os

def get_base_path():
    """
    Get the base path for resources.
    If running as a PyInstaller executable, this is sys._MEIPASS.
    Otherwise, it is the project root directory.
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return sys._MEIPASS
    else:
        # Running from source
        # This file is in dm_toolkit/utils/paths.py
        # Project root is ../../../
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        return project_root

def get_resource_path(relative_path):
    """
    Get the absolute path to a resource.
    """
    base_path = get_base_path()
    return os.path.join(base_path, relative_path)

def ensure_bin_in_path():
    """
    Add the bin directory to sys.path to allow importing the C++ extension.
    """
    if getattr(sys, 'frozen', False):
        # In frozen mode, extension should be in the bundle root or adjacent
        return

    base_path = get_base_path()
    bin_dir = os.path.join(base_path, "bin")
    if os.path.exists(bin_dir) and bin_dir not in sys.path:
        sys.path.append(bin_dir)
