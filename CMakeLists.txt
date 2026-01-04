cmake_minimum_required(VERSION 3.14)

project(DM_AI_Simulator
    VERSION 1.0
    DESCRIPTION "Duel Masters AI Simulator with C++20 Engine and Python AI"
    LANGUAGES CXX
)

# --- Compiler Settings ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Suppress CMake deprecation warnings (specifically for pybind11's older CMake requirement)
set(CMAKE_WARN_DEPRECATED OFF)
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
        add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
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

# Ensure Python development headers are available for native components that call
# into the Python C API (e.g. GIL handling in python_batch_inference.cpp).
find_package(Python COMPONENTS Interpreter Development QUIET)
if(Python_FOUND)
    include_directories(${Python_INCLUDE_DIRS})
else()
    message(STATUS "Python development headers not found via find_package(Python). pybind11 will still provide Python interop for the module target.")
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

    # Fix for MinGW/Clang on Windows when using PyTorch (which adds MSVC flags)
    if(WIN32 AND NOT MSVC)
        message(STATUS "Removing MSVC flags from CMAKE_CXX_FLAGS for MinGW/Clang compatibility")
        foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            if(${flag_var})
                string(REPLACE "/permissive-" "" ${flag_var} "${${flag_var}}")
                string(REPLACE "/EHsc" "" ${flag_var} "${${flag_var}}")
                string(REPLACE "/bigobj" "" ${flag_var} "${${flag_var}}")
                string(REPLACE "/MD" "" ${flag_var} "${${flag_var}}")
                string(REPLACE "/MDd" "" ${flag_var} "${${flag_var}}")
            endif()
        endforeach()

        # Remove flags from torch targets
        foreach(tgt torch torch_cpu torch_cuda c10)
            if(TARGET ${tgt})
                # MSVC flags are often wrapped in generator expressions, making them hard to remove with list(REMOVE_ITEM)
                # Since we are on MinGW, we likely don't need any of the MSVC specific options provided by Torch.
                # Clearing them is the safest bet to avoid "unknown argument" errors.
                set_target_properties(${tgt} PROPERTIES INTERFACE_COMPILE_OPTIONS "")
                message(STATUS "Cleared INTERFACE_COMPILE_OPTIONS for target ${tgt} to remove MSVC flags")
            endif()
        endforeach()
    endif()
else()
    message(STATUS "LibTorch disabled (USE_LIBTORCH=OFF)")
endif()

# Optional: ONNX Runtime for C++ inference (Phase 4.2)
option(USE_ONNXRUNTIME "Enable ONNX Runtime for C++ inference" ON) # Default ON for this task
if(USE_ONNXRUNTIME)
    message(STATUS "ONNX Runtime enabled")
    include(FetchContent)

    if(WIN32)
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-win-x64-1.18.0.zip")
    elseif(APPLE)
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-universal2-1.18.0.tgz")
    else()
        # Linux x64
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz")
    endif()

    FetchContent_Declare(
        onnxruntime_pkg
        URL ${ORT_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime_pkg)

    # Set include and link directories
    # fetchcontent_populate populates onnxruntime_pkg_SOURCE_DIR
    set(ORT_INCLUDE_DIR ${onnxruntime_pkg_SOURCE_DIR}/include)
    set(ORT_LIB_DIR ${onnxruntime_pkg_SOURCE_DIR}/lib)

    include_directories(${ORT_INCLUDE_DIR})
    link_directories(${ORT_LIB_DIR})

    if(WIN32)
        set(ORT_LIBS onnxruntime)
    else()
        set(ORT_LIBS onnxruntime)
    endif()

    add_compile_definitions(USE_ONNXRUNTIME)
else()
    message(STATUS "ONNX Runtime disabled (USE_ONNXRUNTIME=OFF)")
endif()

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
    src/core/game_state.cpp
    src/core/game_state_stats.cpp
    src/core/game_state_tracking.cpp
    src/core/game_state_hash.cpp
    src/core/game_state_loop.cpp
    src/core/game_state_observer.cpp # Added
    src/core/action.hpp
)

# Engine (Phase 2)
set(SRC_ENGINE
    src/engine/actions/intent_generator.cpp
    src/engine/actions/strategies/pending_strategy.cpp
    src/engine/actions/strategies/stack_strategy.cpp
    src/engine/actions/strategies/phase_strategies.cpp
    src/engine/systems/flow/phase_manager.cpp
    src/engine/systems/mana/mana_system.cpp
    src/engine/systems/breaker/breaker_system.cpp # Added
    src/engine/utils/determinizer.cpp
    src/engine/utils/dev_tools.cpp
    src/engine/systems/card/card_registry.cpp
    src/engine/systems/card/effect_system.cpp
    src/engine/systems/card/selection_system.cpp
    src/engine/systems/card/condition_system.cpp
    src/engine/systems/card/json_loader.cpp
    src/engine/game_instance.cpp
    src/engine/systems/mana/cost_payment_system.cpp
    src/engine/systems/trigger_system/trigger_manager.cpp
    src/engine/systems/trigger_system/trigger_system.cpp
    src/engine/game_command/commands.cpp
    src/engine/game_command/action_commands.cpp # Added
    src/engine/systems/pipeline_executor.cpp
    src/engine/systems/command_system.cpp
    src/engine/systems/game_logic_system.cpp
    src/engine/systems/restriction_system.cpp
    src/engine/systems/continuous_effect_system.cpp
    src/engine/systems/breaker/breaker_system.cpp
)

# Utils (Phase 1)
set(SRC_UTILS
    # src/utils/rng.cpp
)

# AI Bridge (Phase 3 & 4)
set(SRC_AI
    src/ai/encoders/tensor_converter.cpp
    src/ai/encoders/token_converter.cpp
    src/ai/encoders/action_encoder.cpp
    src/ai/mcts/mcts.cpp
    src/ai/evaluator/heuristic_evaluator.cpp
    src/ai/evaluator/beam_search_evaluator.cpp
    src/ai/self_play/self_play.cpp
    src/ai/self_play/parallel_runner.cpp
    src/ai/agents/heuristic_agent.cpp
    src/ai/data_collection/data_collector.cpp
    src/ai/scenario/scenario_executor.cpp
    src/ai/solver/lethal_solver.cpp
    src/ai/evolution/deck_evolution.cpp
    src/ai/evolution/meta_environment.cpp
    src/bindings/python_batch_inference.cpp
    src/ai/inference/pimc_generator.cpp
    src/ai/inference/deck_inference.cpp
    src/ai/pomdp/pomdp.cpp
    src/ai/neural_net/self_attention.cpp # Added
    # Optional neural evaluator / inference (guarded by USE_LIBTORCH / USE_ONNXRUNTIME)
    $<$<OR:$<BOOL:${USE_LIBTORCH}>,$<BOOL:${USE_ONNXRUNTIME}>>:src/ai/evaluator/neural_evaluator.cpp>
    $<$<BOOL:${USE_LIBTORCH}>:src/ai/inference/torch_model.cpp>
    $<$<BOOL:${USE_ONNXRUNTIME}>:src/ai/inference/onnx_model.cpp>
)

# --- Targets ---

# Main Simulator Library (Object library to avoid singleton duplication in static linking)
add_library(dm_core OBJECT
    ${SRC_CORE}
    ${SRC_ENGINE}
    ${SRC_UTILS}
    ${SRC_AI}
)
# target_link_libraries cannot be used with OBJECT libraries in older CMake,
# but usually it works for interface/public deps in newer versions.
# We set properties directly.
set_target_properties(dm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)


# Python Module (Phase 3)
pybind11_add_module(dm_ai_module
    src/bindings/bindings.cpp
    src/bindings/bind_core.cpp
    src/bindings/bind_engine.cpp
    src/bindings/bind_ai.cpp
)
target_link_libraries(dm_ai_module PRIVATE dm_core)

# Output the extension module to the bin/ directory
set_target_properties(dm_ai_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

if(USE_LIBTORCH)
    target_link_libraries(dm_core PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(dm_core PUBLIC USE_LIBTORCH)
    target_link_libraries(dm_ai_module PRIVATE ${TORCH_LIBRARIES})
endif()

if(USE_ONNXRUNTIME)
    target_link_libraries(dm_core PRIVATE ${ORT_LIBS})
    target_compile_definitions(dm_core PUBLIC USE_ONNXRUNTIME)
    target_link_libraries(dm_ai_module PRIVATE ${ORT_LIBS})
endif()
