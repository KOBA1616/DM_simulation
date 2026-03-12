# CMake generated Testfile for 
# Source directory: C:/Users/ichirou/DM_simulation
# Build directory: C:/Users/ichirou/DM_simulation/build-ninja
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(python_pytest "C:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe" "-m" "pytest" "-q")
set_tests_properties(python_pytest PROPERTIES  WORKING_DIRECTORY "C:/Users/ichirou/DM_simulation" _BACKTRACE_TRIPLES "C:/Users/ichirou/DM_simulation/CMakeLists.txt;298;add_test;C:/Users/ichirou/DM_simulation/CMakeLists.txt;0;")
subdirs("_deps/pybind11-build")
