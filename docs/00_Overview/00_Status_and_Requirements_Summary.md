# Requirement Definition 00: Status and Requirements Summary

## 1. Project Overview
**Goal:** Develop a Duel Masters AI/Agent system with a C++ engine and Python bindings.
**Current Phase:** Phase 5 (C++ Feature Extraction & Parallelization)
**Focus:** Robustness, Verifiability, Risk Management.

## 2. Current Status
*   **Engine:** C++20 core with `GameState`, `ActionGenerator`, `EffectResolver`.
*   **Python Bindings:** `pybind11` integration (`dm_ai_module`).
*   **AI:** AlphaZero-style MCTS (`ParallelRunner`) + PyTorch training (`train_simple.py`).
*   **GUI:** Python `tkinter` based (`app.py`, `card_editor.py`).
*   **Data:** JSON-based card definitions (`data/cards.json`).

### 2.1. Implemented Features (Recent)
*   **Bounce (Return to Hand):** Implemented `RETURN_TO_HAND` action type in C++ engine and GUI templates.
*   **Deck Search/Look:** Implemented `SEARCH_DECK_BOTTOM` (Look at top N, Add selected to hand, Return rest to bottom).
*   **Mekraid:** Implemented `MEKRAID` (Look at top 3, Play condition, Return rest to bottom).
*   **GUI Editor:** Updated `card_editor.py` with templates for the above effects.

## 3. Requirements (Active)

### 3.1. GUI & Card Effects Expansion
*   **Look at Top Deck (Refinement):**
    *   **Private vs Public:** Ensure visualization differentiates between private info (Executor only) and public info (Both players). *Implementation note: `SEARCH_DECK_BOTTOM` handles the logic, but frontend visualization of "revealed" cards needs verification.*

### 3.2. Core Features (Existing)
*   **Loop Detection:** Hash-based state tracking to prevent infinite loops (Implemented).
*   **Performance:** Move loop detection and heavy logic to C++ (In Progress).
*   **Data Driven:** Use JSON for all card logic.

## 4. Known Issues / Risks
*   **Lethal Puzzle:** Current AI fails `lethal_puzzle_easy` (0% WR). Optimization target.
*   **Complex Effects:** Multi-step effects (Search, Shield Trigger options) need robust handling in C++.

## 5. Next Steps
1.  Verify AI performance with new effects (if applicable to scenarios).
2.  Address Lethal Puzzle performance (0% WR).
3.  Continue Phase 5 C++ feature extraction.
