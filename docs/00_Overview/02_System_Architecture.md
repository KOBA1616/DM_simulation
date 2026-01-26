# System Architecture

## 1. Directory Structure

```
duel_masters_ai/
├── src/                # C++ source code
│   ├── core/           # Core data structures (GameState, Card, Player)
│   ├── engine/         # Game engine logic
│   │   ├── game/       # Game loop and phase management
│   │   ├── systems/    # Subsystems (Mana, Battle, Trigger)
│   │   └── commands/   # GameCommand definitions (Mutate, Flow, etc.)
│   ├── ai/             # AI agents (MCTS, Inference)
│   └── bindings/       # Python bindings (pybind11)
├── data/               # Data files
│   ├── cards.json      # Card DB (JSON format)
├── python/             # Python scripts
│   ├── training/       # Training and Evolution scripts
│   ├── gui/            # Card Editor and Visualization Tools
│   └── tests/          # Python unit tests
└── build/              # Build artifacts
```

## 2. Core Modules

### 2.1 Game State (`src/core`)
- **GameState**: Holds the entire state of the game (players, zones, turn info).
- **Player**: Represents a player (hand, mana, shields, battle zone).
- **CardInstance**: Dynamic card state with `owner` tracking.
- **CardDefinition**: Static card data loaded from JSON.

### 2.2 Game Engine (`src/engine`)
The engine utilizes a **Command-Based Architecture**.
- **PhaseManager**: Manages the game loop and phase transitions.
- **GameCommand**: Atomic instructions (`TRANSITION`, `MUTATE`, `FLOW`, etc.) that modify the game state.
- **EffectSystem**: Compiles and resolves card effects into executable logic, replacing the legacy `EffectResolver`.
- **ConditionSystem**: Evaluates complex conditions (`ConditionDef`) for triggers and effects.
- **TriggerManager**: Event-driven system for handling passive effects and triggers.

### 2.3 AI (`src/ai`)
- **MCTS**: Monte Carlo Tree Search implementation compatible with AlphaZero.
- **ParallelRunner**: Multi-threaded environment for high-speed self-play and evaluation.
- **Inference**: Modules (`DeckInference`, `PimcGenerator`) for handling imperfect information.
- **Evaluator**: Interfaces for state evaluation (`BeamSearchEvaluator`, `NeuralEvaluator`).

## 3. Python Integration
- **dm_ai_module**: The compiled C++ module exposed to Python.
- **Training Loop**: Python scripts drive the learning process, calling C++ for heavy computation (self-play).
- **GUI**: PyQt6 application for data editing and visualizing C++ game states.
