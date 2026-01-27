cmake_minimum_required(VERSION 3.15)
project(index_to_command_native LANGUAGES CXX)
include(FetchContent)

FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.9.2
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(index_to_command_native src/index_to_command.cpp)
pybind11_add_module(command_encoder_native src/command_encoder.cpp)
target_compile_features(index_to_command_native PRIVATE cxx_std_17)
