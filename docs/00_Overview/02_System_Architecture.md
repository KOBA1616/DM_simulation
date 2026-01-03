# System Architecture

## 1. Directory Structure

```
duel_masters_ai/
├── src/                # C++ source code
│   ├── core/           # Core data structures (GameState, Card, Player)
│   ├── engine/         # Game engine logic (PhaseManager, ManaSystem)
│   ├── ai/             # AI agents (MCTS, AlphaZero)
│   └── bindings/       # Python bindings (pybind11)
├── data/               # Data files
│   ├── cards.json              # Card DB (JSON format)
├── python/             # Python scripts
│   ├── training/       # Training scripts
│   └── tests/          # Python unit tests
└── build/              # Build artifacts
```

## 2. Core Modules

### 2.1 Game State (src/core)
- **GameState**: Holds the entire state of the game (players, zones, turn info).
- **Player**: Represents a player (hand, mana, shields, battle zone).
- **Card**: Represents a card instance.
- **CardDefinition**: Static card data (name, cost, power, effects).

### 2.2 Game Engine (src/engine)
- **PhaseManager**: Manages the game loop and phase transitions.
- **ManaSystem**: Handles mana payment and tapping.
- **ActionGenerator**: Generates legal actions for a given state.
- **EffectResolver**: Resolves card effects.

### 2.3 AI (src/ai)
- **MCTS**: Monte Carlo Tree Search implementation.
- **AlphaZero**: Neural network for policy and value estimation.
- **DataCollector**: Collects self-play data for training.

## 3. Python Integration
- **dm_ai_module**: The compiled C++ module exposed to Python.
- **gym interface**: The environment follows the OpenAI Gym interface for RL.
