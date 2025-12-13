import os
import sys
import shutil

# Ensure the build directory and MinGW bin directory are in the DLL search path on Windows
if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    # Add build directory (assuming dm_toolkit is in project_root/dm_toolkit)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    build_path = os.path.join(project_root, 'build')
    
    if os.path.exists(build_path):
        try:
            os.add_dll_directory(build_path)
        except Exception:
            pass

    # Add MinGW bin directory (find c++.exe)
    cpp_path = shutil.which('c++.exe')
    if cpp_path:
        mingw_bin = os.path.dirname(cpp_path)
        try:
            os.add_dll_directory(mingw_bin)
        except Exception:
            pass
