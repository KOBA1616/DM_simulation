# CMake generated Testfile for 
# Source directory: C:/Users/ichirou/DM_simulation
# Build directory: C:/Users/ichirou/DM_simulation/build-msvc
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(python_pytest "C:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe" "-m" "pytest" "-q")
  set_tests_properties(python_pytest PROPERTIES  WORKING_DIRECTORY "C:/Users/ichirou/DM_simulation" _BACKTRACE_TRIPLES "C:/Users/ichirou/DM_simulation/CMakeLists.txt;284;add_test;C:/Users/ichirou/DM_simulation/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(python_pytest "C:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe" "-m" "pytest" "-q")
  set_tests_properties(python_pytest PROPERTIES  WORKING_DIRECTORY "C:/Users/ichirou/DM_simulation" _BACKTRACE_TRIPLES "C:/Users/ichirou/DM_simulation/CMakeLists.txt;284;add_test;C:/Users/ichirou/DM_simulation/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(python_pytest "C:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe" "-m" "pytest" "-q")
  set_tests_properties(python_pytest PROPERTIES  WORKING_DIRECTORY "C:/Users/ichirou/DM_simulation" _BACKTRACE_TRIPLES "C:/Users/ichirou/DM_simulation/CMakeLists.txt;284;add_test;C:/Users/ichirou/DM_simulation/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(python_pytest "C:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe" "-m" "pytest" "-q")
  set_tests_properties(python_pytest PROPERTIES  WORKING_DIRECTORY "C:/Users/ichirou/DM_simulation" _BACKTRACE_TRIPLES "C:/Users/ichirou/DM_simulation/CMakeLists.txt;284;add_test;C:/Users/ichirou/DM_simulation/CMakeLists.txt;0;")
else()
  add_test(python_pytest NOT_AVAILABLE)
endif()
subdirs("_deps/pybind11-build")
