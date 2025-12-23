# Windows Build and Distribution Guide

## Prerequisites

1.  **Python 3.10+**: Ensure Python is installed and added to PATH.
2.  **CMake**: Install CMake (3.14+).
3.  **Compiler**:
    *   **Recommended**: Visual Studio 2019/2022 with C++ Desktop Development (MSVC).
    *   *Alternative*: MinGW-w64 (but might require tweaking for PyTorch/ONNX compatibility).
4.  **Dependencies**:
    *   Install Python deps: `pip install -r requirements.txt`

## Building the C++ Core

1.  Open Command Prompt or PowerShell.
2.  Navigate to the project root.
3.  Create a build directory:
    ```cmd
    mkdir build
    cd build
    ```
4.  Configure CMake:
    *   For MSVC (Standard):
        ```cmd
        cmake ..
        ```
    *   If you need to specify generator:
        ```cmd
        cmake .. -G "Visual Studio 17 2022" -A x64
        ```
5.  Build:
    ```cmd
    cmake --build . --config Release
    ```
6.  Install/Copy Artifacts:
    *   The `dm_ai_module.pyd` file will be in `build/Release/`.
    *   Copy it to the `bin/` folder in the project root:
        ```cmd
        mkdir ..\bin
        copy Release\dm_ai_module.pyd ..\bin\
        ```
    *   **Note**: If you enabled ONNX Runtime (default), you also need `onnxruntime.dll` in the same folder as the `.pyd` or in your system PATH. The build usually downloads it to `build/_deps/onnxruntime_pkg-src/lib/`.

## Running the Application

Use the provided batch script:

```cmd
scripts\run_app.bat
```

Or manually:

```cmd
set PYTHONPATH=%CD%;%CD%\bin
python dm_toolkit\gui\app.py
```

## Packaging for Distribution (PyInstaller)

To create a standalone `.exe`:

1.  Install PyInstaller: `pip install pyinstaller`
2.  Run the build command (ensure `dm_ai_module.pyd` and `onnxruntime.dll` are in `bin/`):

```cmd
pyinstaller --name "DM_Simulator" --windowed ^
    --add-data "data;data" ^
    --add-data "bin/dm_ai_module.pyd;." ^
    --add-data "bin/onnxruntime.dll;." ^
    --icon "docs/icon.ico" ^
    dm_toolkit/gui/app.py
```

*Note: You may need to adjust the `--add-data` separator (`;` for Windows, `:` for Linux).*

The output will be in `dist/DM_Simulator/`.
