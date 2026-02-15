# Game Engine Architecture Refactor

## Overview
This document outlines the re-architecture of the game engine to separate concerns, decouple state from logic, and establish a clear command translation layer.

## Core Components

### 1. GameInstance (Owner)
*   **Role**: Top-level container for a single game session.
*   **Responsibility**:
    *   Holds `GameState` (Data).
    *   Holds `PipelineExecutor` (Engine).
    *   Holds Logic Systems (Phase, Battle, etc.).
    *   Entry point for external consumers (AI, GUI) via `step()`.

### 2. GameState (Data)
*   **Role**: Pure data container.
*   **Responsibility**:
    *   Stores zones, players, turn count, history.
    *   **NO** execution logic (no `active_pipeline`).
    *   **NO** self-driving logic.

### 3. PipelineExecutor (Engine)
*   **Role**: Stack-based virtual machine for game operations.
*   **Responsibility**:
    *   Executes `Instruction`s.
    *   Manages the execution stack (for interrupts like Shield Triggers).
    *   **Stateless** regarding the game (accepts `GameState&` as argument).

### 4. CommandSystem (Translator)
*   **Role**: Converts high-level intent into low-level instructions.
*   **Responsibility**:
    *   Input: `CommandDef` (e.g., "Attack with creature X").
    *   Output: `std::vector<Instruction>` (e.g., "Tap X", "Set attack target", "Process triggers").
    *   Does **NOT** modify `GameState` directly (legacy methods to be deprecated).

## Logic Layer (Systems)

### GameLogicSystem (Director)
*   **Role**: Coordinator of the game flow.
*   **Responsibility**:
    *   Decides *when* to transition phases.
    *   Delegates specific logic to sub-systems.

### PhaseSystem
*   **Role**: Phase management.
*   **Responsibility**:
    *   Start/End of turn procedures.
    *   Phase transitions (Draw -> Mana -> Main...).
    *   Resetting turn-based flags (summoning sickness, mana usage).

### BattleSystem
*   **Role**: Combat logic.
*   **Responsibility**:
    *   Attack declaration checks.
    *   Blocking declaration checks.
    *   Battle resolution (Power comparison, Slayer ability).
    *   Destruction processing.

### ShieldSystem
*   **Role**: Shield mechanics.
*   **Responsibility**:
    *   Breaking shields.
    *   S-Trigger checks and queuing.
    *   Shield addition/removal.

### PlaySystem
*   **Role**: Card usage logic.
*   **Responsibility**:
    *   Playing cards (Creatures/Spells).
    *   Mana payment (Tapping mana).
    *   Evolution source selection.
    *   God Link (future).

### TriggerSystem
*   **Role**: Event observation.
*   **Responsibility**:
    *   Watches for game events (Enter Zone, Destroy, etc.).
    *   Queues `PendingEffect`s.
    *   Does **NOT** execute the effects itself (delegates to Command/Executor).

## Data Flow

1.  **Input**: External agent calls `GameInstance::step()` or `apply_move(CommandDef)`.
2.  **Logic**: `GameLogicSystem` determines valid next steps or processes the input.
3.  **Delegation**: Logic delegates to a specific system (e.g., `BattleSystem::handle_attack`).
4.  **Translation**: System calls `CommandSystem::translate(CommandDef)` to get instructions.
5.  **Execution**: System passes instructions to `PipelineExecutor::execute(state, instructions)`.
6.  **Update**: `PipelineExecutor` modifies `GameState`.
