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
*   **Speed Attacker / Evolution Logic:** Fixed engine to allow creatures with `Speed Attacker` or `Evolution` to attack immediately (ignoring summoning sickness).
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

### 3.3. Future Feature Backlog (User Requested)
*   **System & Engine:**
    *   **Zone Movement Args:** Expand arguments for zone movement functions (ゾーン移動関数の引数を拡充).
    *   **Draw Monitor:** Monitor draw count per turn (各ターンでのドロー枚数の監視).
    *   **Cost Reduction:** Generic cost reduction functions (コスト軽減用汎用関数の実装).
    *   **Battle Zone Checks:** Reference specific costs in Battle Zone (バトルゾーンの特定コストを参照する).
    *   **Graveyard Logic:** Logic for "When placed in graveyard, if it was in BZ (including under evo)" (墓地に置かれた時、バトルゾーンにあれば...).
*   **Card Mechanics:**
    *   **Just Diver:** Hexproof/Untargetable (ジャストダイバー).
    *   **Alternative Cost:** G-Zero, Sympathy, etc. (代替コスト).
    *   **Meteorburn:** Evolution source cost (メテオバーン).
    *   **Neo Evolution:** Creature/Evolution hybrid (ネオ進化カード).
    *   **Ninja Strike:** (ニンジャストライク).
    *   **Unblockable (Temp):** Attack unblockable with duration (攻撃時ブロック禁止効果の実装(効果期限)).
    *   **Global Removal:** Board wipe (全体除去).
    *   **Attack Restriction:** (攻撃制限).
    *   **Anti-Cheat (Meta):** Counters to cost cheating (踏み倒しメタ).
    *   **Mana Recovery:** (マナ回収).
    *   **Reanimation:** (蘇生).
    *   **Modal Effects:** Choose 1 of N (モード).
*   **Tools:**
    *   **GUI Extension:** Expand Card Creation GUI (カード作成GUIの拡張).

## 4. Known Issues / Risks
*   **Complex Effects:** Multi-step effects (Search, Shield Trigger options) need robust handling in C++.
*   **Memory Usage:** High simulation counts in `verify_performance.py` may cause memory allocation errors.

## 5. Next Steps
1.  Verify AI performance with new effects (if applicable to scenarios).
2.  Continue Phase 5 C++ feature extraction (Focus on performance and memory stability).
