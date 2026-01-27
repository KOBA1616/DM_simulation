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
```
