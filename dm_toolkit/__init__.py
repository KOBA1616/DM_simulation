import os
import sys
import shutil

# Ensure native dependencies resolve on Windows.
# Keep this conservative by default for clone distribution; enable dev-only additions
# by setting DM_SIMULATION_DEV=1.
if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Dev convenience: allow loading DLLs from build/ when explicitly enabled.
    if os.environ.get('DM_SIMULATION_DEV', '').strip() == '1':
        build_path = os.path.join(project_root, 'build')
        if os.path.exists(build_path):
            try:
                os.add_dll_directory(build_path)
            except Exception:
                pass

        # MinGW bin directory is a developer environment detail; never rely on it implicitly.
        cpp_path = shutil.which('c++.exe')
        if cpp_path:
            mingw_bin = os.path.dirname(cpp_path)
            try:
                os.add_dll_directory(mingw_bin)
            except Exception:
                pass
