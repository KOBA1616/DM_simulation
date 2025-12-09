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
    # Force UTF-8 source/exec encoding with MSVC
    add_compile_options(/utf-8)
else()
    add_compile_options(-Wall -Wextra -pedantic)
    # Force UTF-8 source/exec encoding on Windows for MinGW/Clang
    if(WIN32)
        add_compile_options(-finput-charset=UTF-8 -fexec-charset=UTF-8)
    endif()
    # Suppress warnings from nlohmann/json with newer Clang
    add_compile_options(-Wno-deprecated-literal-operator)
endif()

# --- Dependencies ---
# Pybind11 is required for Python bindings (Phase 3)
# Try to find an installed pybind11 first; if not present, fetch it via FetchContent
include(FetchContent)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package; fetching via FetchContent")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()
if(NOT pybind11_FOUND)
    # If FetchContent made pybind11 available, mark it as found so downstream
    # commands (e.g., pybind11_add_module) can use the targets. If not found,
    # fail early with a helpful message.
    if(TARGET pybind11::module)
        set(pybind11_FOUND TRUE)
    else()
        message(FATAL_ERROR "pybind11 not found and FetchContent did not provide it.\nEnsure network access or install pybind11 via package manager.")
    endif()
endif()

# Optional: LibTorch for C++ inference (Phase 3.3)
option(USE_LIBTORCH "Enable LibTorch-based C++ inference (optional)" OFF)
if(USE_LIBTORCH)
    find_package(Torch REQUIRED)
    message(STATUS "LibTorch enabled")
else()
    message(STATUS "LibTorch disabled (USE_LIBTORCH=OFF)")
endif()

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

# Optional: user-provided include directory for custom libraries
if(DEFINED ENV{MYLIB_INCLUDE})
    message(STATUS "Including custom library path: $ENV{MYLIB_INCLUDE}")
    include_directories($ENV{MYLIB_INCLUDE})
endif()

# --- Source Files ---
# Define source files variables for better organization

# Core (Phase 1)
set(SRC_CORE
    src/core/types.hpp
    src/core/constants.hpp
    src/core/card_def.hpp
    src/core/game_state.hpp
    src/core/game_state_stats.cpp
    src/core/game_state_tracking.cpp
    src/core/game_state_hash.cpp
    src/core/game_state_loop.cpp
    src/core/action.hpp
)

# Engine (Phase 2)
set(SRC_ENGINE
    src/engine/actions/action_generator.cpp
    src/engine/actions/strategies/pending_strategy.cpp
    src/engine/actions/strategies/stack_strategy.cpp
    src/engine/actions/strategies/phase_strategies.cpp
    src/engine/systems/flow/phase_manager.cpp
    src/engine/effects/effect_resolver.cpp
    src/engine/systems/mana/mana_system.cpp
    src/engine/utils/determinizer.cpp
    src/engine/utils/dev_tools.cpp
    src/engine/systems/card/card_registry.cpp
    src/engine/systems/card/generic_card_system.cpp
    src/engine/systems/card/json_loader.cpp
    src/engine/game_instance.cpp
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
    src/ai/evaluator/neural_evaluator.cpp
    src/ai/evaluator/beam_search_evaluator.cpp
    src/ai/self_play/self_play.cpp
    src/ai/self_play/parallel_runner.cpp
    src/ai/agents/heuristic_agent.cpp
    src/ai/data_collection/data_collector.cpp
    src/ai/scenario/scenario_executor.cpp
    src/ai/solver/lethal_solver.cpp
    src/bindings/python_batch_inference.cpp
    # Optional neural evaluator / inference (guarded by USE_LIBTORCH)
    $<$<BOOL:${USE_LIBTORCH}>:src/ai/inference/torch_model.cpp>
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

# Ensure objects in the static library are position-independent so they
# can be linked into a Python extension/shared library on ELF platforms.
set_target_properties(dm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Executable for C++ Testing/Validation (Phase 1 & 2)
add_executable(dm_sim_test
    src/main_test.cpp
)
target_link_libraries(dm_sim_test PRIVATE dm_core)

# Python Module (Phase 3)
pybind11_add_module(dm_ai_module
    src/bindings/bindings.cpp
)
target_link_libraries(dm_ai_module PRIVATE dm_core)

if(USE_LIBTORCH)
    target_link_libraries(dm_core PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(dm_core PUBLIC USE_LIBTORCH)
    target_link_libraries(dm_ai_module PRIVATE ${TORCH_LIBRARIES})
endif()
