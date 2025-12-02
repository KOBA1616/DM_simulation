# Building DM AI Simulator on Windows with MinGW

This document describes how to build the project on Windows using MinGW-w64 and CMake.

## Prerequisites

1.  **MinGW-w64**: Ensure MinGW-w64 is installed.
    *   Default path assumed: `C:\Program Files (x86)\mingw64`
    *   If installed elsewhere, update the paths in the commands below.
2.  **CMake**: Version 3.14 or higher.
3.  **Python**: Python 3.8+ (3.12 recommended).
4.  **PowerShell**: Recommended shell.

## 1. Environment Setup

We have provided a script to set up the environment variables for MinGW.

Run the following command in PowerShell to add MinGW to your PATH (if not already added):

```powershell
.\scripts\setup_mingw_env.ps1 -GccPath "C:\Program Files (x86)\mingw64\bin\x86_64-w64-mingw32-gcc.exe"
```

**Note**: You may need to restart your terminal for the PATH changes to take effect permanently. For the current session, you can manually set the path:

```powershell
$env:Path = "C:\Program Files (x86)\mingw64\bin;" + $env:Path
```

## 2. Build Instructions

### Step 1: Configure CMake

We use a custom toolchain file `cmake/toolchain_mingw.cmake` to ensure CMake uses the correct MinGW compilers. We also need to tell CMake where `pybind11` is located.

```powershell
# Set Path for current session
$env:Path = "C:\Program Files (x86)\mingw64\bin;" + $env:Path

# Get pybind11 path from python
$pybind_dir = (python -m pybind11 --cmakedir).Trim()

# Configure
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/toolchain_mingw.cmake" -DCMAKE_PREFIX_PATH="$pybind_dir"
```

### Step 2: Build

```powershell
cmake --build build
```

This will create:
*   `build/dm_core.lib` (Static library)
*   `build/dm_sim_test.exe` (C++ Test Executable)
*   `build/dm_ai_module.cp312-win_amd64.pyd` (Python Module)

## 3. Running Tests

### C++ Tests

```powershell
.\build\dm_sim_test.exe
```

### Python Binding Tests

To run Python scripts that import the C++ module, you need to:
1.  Add the `build` directory to `PYTHONPATH`.
2.  Ensure MinGW DLLs are accessible (added to PATH or via `os.add_dll_directory`).

We have updated `python/scripts/test_binding.py` to handle DLL loading automatically.

```powershell
$env:Path = "C:\Program Files (x86)\mingw64\bin;" + $env:Path
$env:PYTHONPATH = "$PWD/build"
python python/scripts/test_binding.py
```

## Troubleshooting

*   **CMake Error: Could not find toolchain file**: Ensure you are using the absolute path or correct relative path to `cmake/toolchain_mingw.cmake`.
*   **ImportError: DLL load failed**: Make sure `C:\Program Files (x86)\mingw64\bin` is in your system PATH or added via `os.add_dll_directory` in your Python script.
*   **FetchContent Error**: We have switched to a local vendor copy of `nlohmann/json` in `third_party/` to avoid CMake policy issues. Ensure `third_party/nlohmann/json.hpp` exists.
