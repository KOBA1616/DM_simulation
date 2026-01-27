cmake_minimum_required(VERSION 3.15)
project(index_to_command_native LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(index_to_command_native src/index_to_command.cpp)
target_compile_features(index_to_command_native PRIVATE cxx_std_17)
