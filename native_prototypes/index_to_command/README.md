Build and test the `index_to_command_native` prototype using CMake and pybind11.

Requirements:
- CMake 3.15+
- A C++17 compiler
- pybind11 installed or available to CMake (e.g., `pip install pybind11`)

Build (out-of-source):

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
# The built module will be a shared object named index_to_command_native(.pyd/.so)
```

Python quick test (after build):

```python
import index_to_command_native as native
print(native.index_to_command(0))
print(native.index_to_command(5))
print(native.index_to_command(25))

If you don't build the native module, use the provided Python wrapper:

```py
from native_prototypes.index_to_command.index_to_command import index_to_command
print(index_to_command(0))
```

Build (Windows example using PowerShell and CMake):

```powershell
mkdir build && cd build
cmake -S .. -B . -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

Or with Ninja / Unix-like:

```bash
mkdir build && cd build
cmake -S .. -B . -G Ninja
cmake --build .
```

After successful build, the module `index_to_command_native` will be importable from Python.
```
