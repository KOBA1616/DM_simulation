cmake_minimum_required(VERSION 3.14)

project(DM_AI_Simulator
    VERSION 1.0
    DESCRIPTION "Duel Masters AI Simulator with C++20 Engine and Python AI"
    LANGUAGES CXX
)

# --- Compiler Settings ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# --- Dependencies ---
# Pybind11 is required for Python bindings (Phase 3)
find_package(pybind11 REQUIRED)

# nlohmann_json for Generic Card System (Phase 3)
# Using local vendor copy to avoid FetchContent policy issues
# include(FetchContent)
# FetchContent_Declare(
#     json
#     URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
# )
# FetchContent_MakeAvailable(json)

# --- Include Directories ---
include_directories(src)
include_directories(third_party)

# --- Source Files ---
# Define source files variables for better organization

# Core (Phase 1)
set(SRC_CORE
    src/core/types.hpp
    src/core/constants.hpp
    src/core/card_def.hpp
    src/core/game_state.hpp
    src/core/action.hpp
)

# Engine (Phase 2)
set(SRC_ENGINE
    src/engine/action_gen/action_generator.cpp
    src/engine/flow/phase_manager.cpp
    src/engine/effects/effect_resolver.cpp
    src/engine/mana/mana_system.cpp
    src/engine/utils/determinizer.cpp
    src/engine/utils/dev_tools.cpp
    src/engine/card_system/card_registry.cpp
    src/engine/card_system/generic_card_system.cpp
)

# Utils (Phase 1)
set(SRC_UTILS
    src/utils/csv_loader.cpp
    # src/utils/rng.cpp
)

# AI Bridge (Phase 3)
set(SRC_AI
    src/ai/encoders/tensor_converter.cpp
    src/ai/encoders/action_encoder.cpp
    src/ai/mcts/mcts.cpp
    src/ai/evaluator/heuristic_evaluator.cpp
    src/ai/self_play/self_play.cpp
    src/ai/self_play/parallel_runner.cpp
)

# --- Targets ---

# Main Simulator Library (Static library for logic)
add_library(dm_core STATIC
    ${SRC_CORE}
    ${SRC_ENGINE}
    ${SRC_UTILS}
    ${SRC_AI}
)
target_link_libraries(dm_core PUBLIC) # nlohmann_json::nlohmann_json removed (header-only)

# Executable for C++ Testing/Validation (Phase 1 & 2)
add_executable(dm_sim_test
    src/main_test.cpp
)
target_link_libraries(dm_sim_test PRIVATE dm_core)

# Python Module (Phase 3)
pybind11_add_module(dm_ai_module
    src/python/bindings.cpp
)
target_link_libraries(dm_ai_module PRIVATE dm_core)
