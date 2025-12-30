# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-src")
  file(MAKE_DIRECTORY "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-build"
  "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-subbuild/onnxruntime_pkg-populate-prefix"
  "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-subbuild/onnxruntime_pkg-populate-prefix/tmp"
  "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-subbuild/onnxruntime_pkg-populate-prefix/src/onnxruntime_pkg-populate-stamp"
  "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-subbuild/onnxruntime_pkg-populate-prefix/src"
  "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-subbuild/onnxruntime_pkg-populate-prefix/src/onnxruntime_pkg-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-subbuild/onnxruntime_pkg-populate-prefix/src/onnxruntime_pkg-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/ichirou/DM_simulation/build_vs/_deps/onnxruntime_pkg-subbuild/onnxruntime_pkg-populate-prefix/src/onnxruntime_pkg-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
