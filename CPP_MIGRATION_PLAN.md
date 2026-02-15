# Python→C++ 移行実装計画

## 実装状況

### ✅ Phase 1: AI選択ロジックの統一（完了）
- 実装完了日: 2024年
- 詳細レポート: [PHASE1_IMPLEMENTATION_REPORT.md](docs/reports/PHASE1_IMPLEMENTATION_REPORT.md)
- 変更ファイル:
  - ✅ `src/engine/ai/simple_ai.hpp` - 新規作成
  - ✅ `src/engine/ai/simple_ai.cpp` - 新規作成
  - ✅ `src/engine/game_instance.cpp` - SimpleAI使用に更新
  - ✅ `dm_toolkit/gui/game_session.py` - _select_ai_action()削除
  - ✅ `CMakeLists.txt` - simple_ai.cpp追加
- テスト: `test_phase1_simple_ai.py`

### ✅ Phase 2: プレイヤーモード管理C++化（完了）
- 実装完了日: 2024年
- 詳細レポート: [PHASE2_IMPLEMENTATION_REPORT.md](docs/reports/PHASE2_IMPLEMENTATION_REPORT.md)
- 変更ファイル:
  - ✅ `src/core/types.hpp` - PlayerMode enum追加
  - ✅ `src/core/game_state.hpp` - player_modes配列とis_human_player()追加
  - ✅ `src/bindings/bind_core.cpp` - PlayerModeバインディング追加
  - ✅ `src/engine/game_instance.cpp` - Humanプレイヤーチェック追加
  - ✅ `dm_toolkit/gui/game_session.py` - GameState.player_modes使用に移行
  - ✅ `dm_toolkit/gui/app.py` - is_human_player()使用に更新
- テスト: `test_phase2_player_modes.py`
- ビルドスクリプト: `build_and_test_phase2.ps1`

### 🔄 Phase 3: イベント通知システム（未着手）
- ステータス: 計画段階

### 🔄 Phase 4: 自動進行スレッド化（未着手）
- ステータス: 計画段階

### 🔄 Phase 5: レガシーラッパー削除（未着手）
- ステータス: 計画段階

---

## 概要
本ドキュメントでは、[GAME_STARTUP_FLOW_ANALYSIS.md](GAME_STARTUP_FLOW_ANALYSIS.md)で特定したPython側のゲーム進行管理処理を、段階的にC++化する具体的な実装計画を提示します。

---

## 実装優先順位マトリクス

| 項目 | 複雑度 | 効果 | 優先度 | 期間 | 状態 |
|-----|--------|------|--------|------|------|
| AI選択ロジック統一 | 低 | 中 | 🔴 高 | 1-2日 | ✅ 完了 |
| プレイヤーモード管理C++化 | 低 | 中 | 🔴 高 | 1日 | ✅ 完了 |
| イベント通知システム | 高 | 高 | 🟡 中 | 2-3日 | ⏸️ 未着手 |
| 自動進行スレッド化 | 中 | 高 | 🟡 中 | 2-3日 | ⏸️ 未着手 |
| レガシーラッパー削除 | 高 | 高 | 🟢 低 | 3-5日 | ⏸️ 未着手 |

---

## Phase 1: AI選択ロジックの統一（Priority 1）

### 目的
現在Python側とC++側で重複実装されているAI選択ロジックを、C++に統一する。

### 現状分析

#### Python側の実装
**ファイル**: [dm_toolkit/gui/game_session.py](dm_toolkit/gui/game_session.py#L291-L330)

```python
def _select_ai_action(self, cmds: List[Any]) -> Any:
    # 優先度1: RESOLVE_EFFECT
    for cmd in cmds:
        if cmd.to_dict().get('type') == 'RESOLVE_EFFECT':
            return cmd
    
    # 優先度2: PLAY_FROM_ZONE
    for cmd in cmds:
        if cmd.to_dict().get('type') == 'PLAY_FROM_ZONE':
            return cmd
    
    # 優先度3: 非PASS
    for cmd in cmds:
        if cmd.to_dict().get('type') != 'PASS':
            return cmd
    
    # 優先度4: PASS
    return cmds[0] if cmds else None
```

#### C++側の実装
**ファイル**: [src/engine/game_instance.cpp](src/engine/game_instance.cpp#L105-L165)

```cpp
bool GameInstance::step() {
    // 優先度1: RESOLVE_EFFECT
    for (const auto& a : actions) {
        if (a.type == PlayerIntent::RESOLVE_EFFECT) {
            selected = &a;
            break;
        }
    }
    
    // 優先度2: PLAY_CARD
    if (!selected) {
        for (const auto& a : actions) {
            if (a.type == PlayerIntent::PLAY_CARD) {
                selected = &a;
                break;
            }
        }
    }
    
    // 優先度3-6: ATTACK, MANA_CHARGE, その他, PASS
    // ...
}
```

### 問題点
1. 同じロジックが2箇所に存在
2. Python側は使用されていない（step_game()ではC++のstep()を呼んでいる）
3. 保守性が低い

### 実装タスク

#### タスク1.1: C++側の選択ロジック分離

**新規ファイル**: `src/engine/ai/simple_ai.hpp`

```cpp
#pragma once
#include "core/game_state.hpp"
#include "core/action.hpp"
#include <vector>
#include <optional>

namespace dm::engine::ai {

/**
 * Simple priority-based AI for action selection
 */
class SimpleAI {
public:
    /**
     * Select an action based on priority:
     * 1. RESOLVE_EFFECT (must complete pending effects)
     * 2. PLAY_CARD (play cards from hand)
     * 3. ATTACK (attack creatures/player)
     * 4. MANA_CHARGE (in MANA phase)
     * 5. Other actions
     * 6. PASS (exit phase)
     * 
     * @return Index of selected action, or nullopt if no action
     */
    static std::optional<size_t> select_action(
        const std::vector<core::Action>& actions,
        const core::GameState& state
    );

private:
    static int get_priority(const core::Action& action);
};

} // namespace dm::engine::ai
```

**新規ファイル**: `src/engine/ai/simple_ai.cpp`

```cpp
#include "simple_ai.hpp"
#include <algorithm>

namespace dm::engine::ai {

using namespace dm::core;

std::optional<size_t> SimpleAI::select_action(
    const std::vector<Action>& actions,
    const GameState& state
) {
    if (actions.empty()) {
        return std::nullopt;
    }

    // Find action with highest priority
    size_t best_idx = 0;
    int best_priority = get_priority(actions[0]);

    for (size_t i = 1; i < actions.size(); ++i) {
        int priority = get_priority(actions[i]);
        if (priority > best_priority) {
            best_priority = priority;
            best_idx = i;
        }
    }

    return best_idx;
}

int SimpleAI::get_priority(const Action& action) {
    switch (action.type) {
        case PlayerIntent::RESOLVE_EFFECT:
            return 100;  // Highest priority
        
        case PlayerIntent::PLAY_CARD:
        case PlayerIntent::PLAY_CARD_INTERNAL:
            return 80;
        
        case PlayerIntent::ATTACK_PLAYER:
        case PlayerIntent::ATTACK_CREATURE:
            return 60;
        
        case PlayerIntent::MANA_CHARGE:
            return 40;
        
        case PlayerIntent::PASS:
            return 0;   // Lowest priority
        
        default:
            return 20;  // Other actions
    }
}

} // namespace dm::engine::ai
```

#### タスク1.2: GameInstanceの更新

**ファイル**: [src/engine/game_instance.cpp](src/engine/game_instance.cpp)

```cpp
#include "ai/simple_ai.hpp"  // 追加

bool GameInstance::step() {
    // ... (existing code: generate actions)
    
    // OLD: Inline priority selection
    /*
    const Action* selected = nullptr;
    
    for (const auto& a : actions) {
        if (a.type == PlayerIntent::RESOLVE_EFFECT) {
            selected = &a;
            break;
        }
    }
    // ... (more priority checks)
    */
    
    // NEW: Use SimpleAI
    auto idx = ai::SimpleAI::select_action(actions, state);
    
    if (idx.has_value()) {
        resolve_action(actions[*idx]);
        return true;
    }
    
    return false;
}
```

#### タスク1.3: Python側の削除

**ファイル**: [dm_toolkit/gui/game_session.py](dm_toolkit/gui/game_session.py)

```python
# DELETE: _select_ai_action() method (lines 291-330)
# This logic is now fully in C++ (SimpleAI class)
```

#### タスク1.4: CMakeLists.txt更新

**ファイル**: [CMakeLists.txt](CMakeLists.txt)

```cmake
# AI module
set(AI_SOURCES
    src/engine/ai/simple_ai.cpp
)

# Add to dm_ai_module target
add_library(dm_ai_module MODULE
    # ... existing sources ...
    ${AI_SOURCES}
)
```

### テスト計画

**新規ファイル**: `tests/cpp/test_simple_ai.cpp`

```cpp
#include <gtest/gtest.h>
#include "engine/ai/simple_ai.hpp"

TEST(SimpleAI, SelectResolveEffect) {
    std::vector<Action> actions = {
        Action{PlayerIntent::PASS, 0, 0, 0},
        Action{PlayerIntent::RESOLVE_EFFECT, 1, 0, 0},
        Action{PlayerIntent::PLAY_CARD, 2, 0, 0}
    };
    
    GameState state(42);
    auto idx = SimpleAI::select_action(actions, state);
    
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 1);  // RESOLVE_EFFECT selected
}

TEST(SimpleAI, SelectPlayCardWhenNoEffect) {
    std::vector<Action> actions = {
        Action{PlayerIntent::PASS, 0, 0, 0},
        Action{PlayerIntent::PLAY_CARD, 2, 0, 0},
        Action{PlayerIntent::MANA_CHARGE, 3, 0, 0}
    };
    
    GameState state(42);
    auto idx = SimpleAI::select_action(actions, state);
    
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 1);  // PLAY_CARD selected
}

TEST(SimpleAI, EmptyActions) {
    std::vector<Action> actions;
    GameState state(42);
    auto idx = SimpleAI::select_action(actions, state);
    
    EXPECT_FALSE(idx.has_value());
}
```

### 検証手順

```powershell
# 1. ビルド
cmake --build build-msvc --config Release

# 2. C++単体テスト実行
.\build-msvc\tests\Release\dm_tests.exe --gtest_filter=SimpleAI.*

# 3. 統合テスト（GUI起動して自動進行確認）
.\scripts\run_gui.ps1

# 4. ログ確認
Get-Content logs\intent_actions.txt | Select-Object -Last 50
```

### 期待される効果

- **コード削減**: Python側の_select_ai_action()削除（約40行）
- **保守性向上**: ロジックが1箇所に集約
- **拡張性向上**: 将来的に高度なAI（MCTSなど）への切り替えが容易

---

## Phase 2: プレイヤーモード管理のC++化（Priority 2）

### 目的
現在Python側の`GameSession.player_modes`で管理されているプレイヤーモード（Human/AI）を、GameStateに移行する。

### 現状分析

#### Python側の実装
**ファイル**: [dm_toolkit/gui/game_session.py](dm_toolkit/gui/game_session.py#L44)

```python
class GameSession:
    def __init__(self, ...):
        self.player_modes: Dict[int, str] = {0: 'AI', 1: 'AI'}
    
    def set_player_mode(self, player_id: int, mode: str):
        self.player_modes[player_id] = mode
```

**使用箇所**:
- [dm_toolkit/gui/game_session.py#L183](dm_toolkit/gui/game_session.py#L183): `is_human = (self.player_modes.get(active_pid) == 'Human')`
- [dm_toolkit/gui/app.py#L90](dm_toolkit/gui/app.py#L90): `if all(mode == 'AI' for mode in self.session.player_modes.values())`

### 実装タスク

#### タスク2.1: GameStateに追加

**ファイル**: [src/core/game_state.hpp](src/core/game_state.hpp)

```cpp
// Add enum
enum class PlayerMode : uint8_t {
    AI = 0,
    HUMAN = 1
};

struct GameState {
    // ... existing fields ...
    
    // NEW: Player modes
    std::array<PlayerMode, MAX_PLAYERS> player_modes{PlayerMode::AI, PlayerMode::AI};
    
    // Helper method
    bool is_human_player(PlayerID pid) const {
        return player_modes[pid] == PlayerMode::HUMAN;
    }
};
```

#### タスク2.2: PyBind11バインディング追加

**ファイル**: [src/python_bindings/core_bindings.cpp](src/python_bindings/core_bindings.cpp)

```cpp
void init_core_bindings(py::module& m) {
    // Enum binding
    py::enum_<PlayerMode>(m, "PlayerMode")
        .value("AI", PlayerMode::AI)
        .value("HUMAN", PlayerMode::HUMAN);
    
    // GameState binding (add to existing)
    py::class_<GameState>(m, "GameState")
        // ... existing bindings ...
        .def_readwrite("player_modes", &GameState::player_modes)
        .def("is_human_player", &GameState::is_human_player);
}
```

#### タスク2.3: GameInstanceでの使用

**ファイル**: [src/engine/game_instance.cpp](src/engine/game_instance.cpp)

```cpp
bool GameInstance::step() {
    if (state.game_over) return false;
    
    // NEW: Check if current player is human - skip auto-step
    PlayerID active_pid = state.active_player_id;
    if (state.is_human_player(active_pid)) {
        // Human player - step() should not be called
        std::cout << "[step] Human player turn, returning false\n";
        return false;
    }
    
    // AI player - continue with auto-step
    // ... (existing code)
}
```

#### タスク2.4: Python側の移行

**ファイル**: [dm_toolkit/gui/game_session.py](dm_toolkit/gui/game_session.py)

```python
class GameSession:
    def __init__(self, ...):
        # DELETE: self.player_modes: Dict[int, str] = {0: 'AI', 1: 'AI'}
        pass
    
    def set_player_mode(self, player_id: int, mode: str):
        """Set player mode (now delegates to C++ GameState)"""
        if not self.gs:
            return
        
        # Convert string to enum
        if mode == 'Human':
            self.gs.player_modes[player_id] = dm_ai_module.PlayerMode.HUMAN
        else:
            self.gs.player_modes[player_id] = dm_ai_module.PlayerMode.AI
        
        self.callback_log(f"P{player_id} mode set to: {mode}")
    
    def step_game(self):
        # ... existing code ...
        
        active_pid = EngineCompat.get_active_player_id(self.gs)
        
        # NEW: Use GameState property
        is_human = self.gs.is_human_player(active_pid)
        
        # OLD: is_human = (self.player_modes.get(active_pid) == 'Human')
        
        # ... rest of code ...
```

**ファイル**: [dm_toolkit/gui/app.py](dm_toolkit/gui/app.py)

```python
# Line 90 - auto-start check
# OLD:
# if all(mode == 'AI' for mode in self.session.player_modes.values()):

# NEW:
if all(mode == dm_ai_module.PlayerMode.AI for mode in self.session.gs.player_modes):
    self.is_running = True
    self.timer.start(500)
```

### テスト計画

**新規ファイル**: `test_player_mode_cpp.py`

```python
import dm_ai_module

def test_default_player_modes():
    """Both players should default to AI mode"""
    gs = dm_ai_module.GameState(42)
    assert gs.player_modes[0] == dm_ai_module.PlayerMode.AI
    assert gs.player_modes[1] == dm_ai_module.PlayerMode.AI

def test_set_human_mode():
    """Setting human mode should work"""
    gs = dm_ai_module.GameState(42)
    gs.player_modes[0] = dm_ai_module.PlayerMode.HUMAN
    
    assert gs.is_human_player(0) == True
    assert gs.is_human_player(1) == False

def test_game_instance_respects_mode():
    """GameInstance should not auto-step for human players"""
    db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    gi = dm_ai_module.GameInstance(42, db)
    
    # Set P0 as human
    gi.state.player_modes[0] = dm_ai_module.PlayerMode.HUMAN
    
    # step() should return False for human player
    result = gi.step()
    assert result == False, "step() should not execute for human player"
```

### 検証手順

```powershell
# 1. ビルド
cmake --build build-msvc --config Release

# 2. Python単体テスト
python test_player_mode_cpp.py

# 3. GUI動作確認（人間プレイヤーモード）
.\scripts\run_gui.ps1
# → UI上でP0をHumanに設定して、手動操作できることを確認

# 4. AI vs AI確認
# → 両プレイヤーAIのまま自動進行することを確認
```

### 期待される効果

- **状態の完全性**: 全てのゲーム状態がGameStateに集約
- **同期不要**: Python-C++間の同期処理削減
- **セーブ対応**: プレイヤーモードもゲーム状態として保存可能

---

## Phase 3: イベント通知システム構築（Priority 3）

### 目的
C++エンジンからPython GUIへの状態変更通知を、コールバックベースからイベント駆動に移行。

### アーキテクチャ設計

```
┌─────────────────────────────────┐
│       C++ Engine Core            │
│                                  │
│  ┌────────────────────────────┐ │
│  │   EventDispatcher          │ │
│  │                            │ │
│  │  ┌──────────────────────┐ │ │
│  │  │ Event Queue          │ │ │
│  │  │  - STATE_CHANGED     │ │ │
│  │  │  - ACTION_EXECUTED   │ │ │
│  │  │  - PHASE_CHANGED     │ │ │
│  │  │  - INPUT_REQUESTED   │ │ │
│  │  └──────────────────────┘ │ │
│  │                            │ │
│  │  emit(event) ────────────┐│ │
│  └────────────────────────────┘ │
└─────────────────────────────┼───┘
                               │
                  ┌────────────┼────────────┐
                  │  PyBind11 Bridge         │
                  │  (Thread-safe queue)     │
                  └────────────┼────────────┘
                               │
┌─────────────────────────────┼───┐
│         Python GUI           ↓   │
│                                  │
│  game_instance.subscribe(       │
│      EventType.STATE_CHANGED,   │
│      lambda e: update_ui()      │
│  )                               │
└──────────────────────────────────┘
```

### 実装タスク

#### タスク3.1: イベント型定義

**新規ファイル**: `src/engine/events/game_event.hpp`

```cpp
#pragma once
#include "core/game_state.hpp"
#include "core/action.hpp"
#include <variant>
#include <string>

namespace dm::engine::events {

enum class EventType {
    STATE_CHANGED,      // GameState modified
    ACTION_EXECUTED,    // Action was executed
    PHASE_CHANGED,      // Phase transition
    TURN_CHANGED,       // Turn number changed
    INPUT_REQUESTED,    // Waiting for user input
    GAME_OVER,          // Game ended
    LOG_MESSAGE         // Log output
};

struct StateChangedEvent {
    const core::GameState& state;
};

struct ActionExecutedEvent {
    const core::Action& action;
    core::PlayerID executor;
};

struct PhaseChangedEvent {
    core::Phase old_phase;
    core::Phase new_phase;
};

struct InputRequestedEvent {
    core::PlayerID player_id;
    std::string query_type;
};

struct LogMessageEvent {
    std::string message;
    int level;  // 0=info, 1=warning, 2=error
};

using EventData = std::variant<
    StateChangedEvent,
    ActionExecutedEvent,
    PhaseChangedEvent,
    InputRequestedEvent,
    LogMessageEvent
>;

struct GameEvent {
    EventType type;
    EventData data;
    uint64_t timestamp;  // Milliseconds since epoch
};

} // namespace dm::engine::events
```

#### タスク3.2: EventDispatcher実装

**新規ファイル**: `src/engine/events/event_dispatcher.hpp`

```cpp
#pragma once
#include "game_event.hpp"
#include <functional>
#include <vector>
#include <mutex>
#include <queue>

namespace dm::engine::events {

class EventDispatcher {
public:
    using Callback = std::function<void(const GameEvent&)>;
    using CallbackID = size_t;

    EventDispatcher() = default;
    ~EventDispatcher() = default;

    // Thread-safe subscription
    CallbackID subscribe(EventType type, Callback callback);
    void unsubscribe(CallbackID id);

    // Emit event (thread-safe)
    void emit(GameEvent event);

    // Process queued events (call from main thread)
    void process_events();

private:
    struct Subscription {
        CallbackID id;
        EventType type;
        Callback callback;
    };

    std::vector<Subscription> subscriptions_;
    std::queue<GameEvent> event_queue_;
    std::mutex mutex_;
    CallbackID next_id_ = 1;
};

} // namespace dm::engine::events
```

**新規ファイル**: `src/engine/events/event_dispatcher.cpp`

```cpp
#include "event_dispatcher.hpp"
#include <chrono>

namespace dm::engine::events {

EventDispatcher::CallbackID EventDispatcher::subscribe(
    EventType type,
    Callback callback
) {
    std::lock_guard<std::mutex> lock(mutex_);
    CallbackID id = next_id_++;
    subscriptions_.push_back({id, type, std::move(callback)});
    return id;
}

void EventDispatcher::unsubscribe(CallbackID id) {
    std::lock_guard<std::mutex> lock(mutex_);
    subscriptions_.erase(
        std::remove_if(
            subscriptions_.begin(),
            subscriptions_.end(),
            [id](const Subscription& sub) { return sub.id == id; }
        ),
        subscriptions_.end()
    );
}

void EventDispatcher::emit(GameEvent event) {
    // Set timestamp
    auto now = std::chrono::system_clock::now();
    event.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();

    std::lock_guard<std::mutex> lock(mutex_);
    event_queue_.push(std::move(event));
}

void EventDispatcher::process_events() {
    std::queue<GameEvent> local_queue;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::swap(local_queue, event_queue_);
    }

    while (!local_queue.empty()) {
        const auto& event = local_queue.front();
        
        // Dispatch to subscribers
        for (const auto& sub : subscriptions_) {
            if (sub.type == event.type) {
                try {
                    sub.callback(event);
                } catch (...) {
                    // Swallow exceptions to not affect engine
                }
            }
        }
        
        local_queue.pop();
    }
}

} // namespace dm::engine::events
```

#### タスク3.3: GameInstanceへの統合

**ファイル**: [src/engine/game_instance.hpp](src/engine/game_instance.hpp)

```cpp
#include "events/event_dispatcher.hpp"

class GameInstance {
public:
    // ... existing ...
    
    // Event system
    std::shared_ptr<events::EventDispatcher> event_dispatcher() {
        return event_dispatcher_;
    }

private:
    std::shared_ptr<events::EventDispatcher> event_dispatcher_;
};
```

**ファイル**: [src/engine/game_instance.cpp](src/engine/game_instance.cpp)

```cpp
GameInstance::GameInstance(uint32_t seed, ...)
    : state(seed), card_db(db) {
    // ... existing initialization ...
    
    event_dispatcher_ = std::make_shared<events::EventDispatcher>();
}

bool GameInstance::step() {
    // ... existing code ...
    
    if (selected) {
        // Emit event before execution
        events::GameEvent event;
        event.type = events::EventType::ACTION_EXECUTED;
        event.data = events::ActionExecutedEvent{*selected, state.active_player_id};
        event_dispatcher_->emit(event);
        
        resolve_action(*selected);
        
        // Emit state changed event
        events::GameEvent state_event;
        state_event.type = events::EventType::STATE_CHANGED;
        state_event.data = events::StateChangedEvent{state};
        event_dispatcher_->emit(state_event);
        
        return true;
    }
    
    return false;
}

void GameInstance::start_game() {
    PhaseManager::start_game(state, *card_db);
    
    // Emit game started event
    events::GameEvent event;
    event.type = events::EventType::STATE_CHANGED;
    event.data = events::StateChangedEvent{state};
    event_dispatcher_->emit(event);
}
```

#### タスク3.4: PyBind11バインディング

**ファイル**: [src/python_bindings/event_bindings.cpp](src/python_bindings/event_bindings.cpp)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "engine/events/event_dispatcher.hpp"

namespace py = pybind11;
using namespace dm::engine::events;

void init_event_bindings(py::module& m) {
    // EventType enum
    py::enum_<EventType>(m, "EventType")
        .value("STATE_CHANGED", EventType::STATE_CHANGED)
        .value("ACTION_EXECUTED", EventType::ACTION_EXECUTED)
        .value("PHASE_CHANGED", EventType::PHASE_CHANGED)
        .value("INPUT_REQUESTED", EventType::INPUT_REQUESTED)
        .value("GAME_OVER", EventType::GAME_OVER)
        .value("LOG_MESSAGE", EventType::LOG_MESSAGE);
    
    // GameEvent (simplified for Python)
    py::class_<GameEvent>(m, "GameEvent")
        .def_readonly("type", &GameEvent::type)
        .def_readonly("timestamp", &GameEvent::timestamp);
    
    // EventDispatcher
    py::class_<EventDispatcher, std::shared_ptr<EventDispatcher>>(m, "EventDispatcher")
        .def("subscribe", &EventDispatcher::subscribe)
        .def("unsubscribe", &EventDispatcher::unsubscribe)
        .def("process_events", &EventDispatcher::process_events);
}
```

#### タスク3.5: Python側の移行

**ファイル**: [dm_toolkit/gui/game_session.py](dm_toolkit/gui/game_session.py)

```python
class GameSession:
    def __init__(self, ...):
        self.callback_update_ui = callback_update_ui
        self.callback_log = callback_log
        # ... other callbacks ...
        
        # NEW: Event subscriptions (stored for cleanup)
        self._event_subscriptions = []
    
    def initialize_game(self, card_db, seed=42):
        # ... existing initialization ...
        
        # NEW: Subscribe to events
        dispatcher = self.game_instance.event_dispatcher()
        
        # State changed → UI update
        sub_id = dispatcher.subscribe(
            dm_ai_module.EventType.STATE_CHANGED,
            self._on_state_changed
        )
        self._event_subscriptions.append(sub_id)
        
        # Action executed → Log
        sub_id = dispatcher.subscribe(
            dm_ai_module.EventType.ACTION_EXECUTED,
            self._on_action_executed
        )
        self._event_subscriptions.append(sub_id)
        
        # Log message → UI log
        sub_id = dispatcher.subscribe(
            dm_ai_module.EventType.LOG_MESSAGE,
            self._on_log_message
        )
        self._event_subscriptions.append(sub_id)
    
    def _on_state_changed(self, event):
        """Called when game state changes"""
        self.callback_update_ui()
    
    def _on_action_executed(self, event):
        """Called when action is executed"""
        if self.callback_action_executed:
            self.callback_action_executed(None)  # Legacy callback
    
    def _on_log_message(self, event):
        """Called when engine emits log message"""
        # Extract message from event.data
        # self.callback_log(message)
        pass
    
    def step_game(self):
        # ... existing code ...
        
        # AI player - use C++ step()
        success = self.game_instance.step()
        
        # NEW: Process events (must be called from Python main thread)
        self.game_instance.event_dispatcher().process_events()
        
        # OLD: self.callback_update_ui()  ← Now handled by event
```

### テスト計画

**新規ファイル**: `test_event_system.py`

```python
import dm_ai_module

class EventCollector:
    def __init__(self):
        self.events = []
    
    def on_event(self, event):
        self.events.append(event)

def test_state_changed_event():
    """STATE_CHANGED event should be emitted on state modification"""
    db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    gi = dm_ai_module.GameInstance(42, db)
    
    collector = EventCollector()
    gi.event_dispatcher().subscribe(
        dm_ai_module.EventType.STATE_CHANGED,
        collector.on_event
    )
    
    # Trigger state change
    gi.start_game()
    gi.event_dispatcher().process_events()
    
    # Should have received event
    assert len(collector.events) > 0
    assert collector.events[0].type == dm_ai_module.EventType.STATE_CHANGED

def test_action_executed_event():
    """ACTION_EXECUTED event should be emitted on action execution"""
    db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    gi = dm_ai_module.GameInstance(42, db)
    gi.state.setup_test_duel()
    
    collector = EventCollector()
    gi.event_dispatcher().subscribe(
        dm_ai_module.EventType.ACTION_EXECUTED,
        collector.on_event
    )
    
    # Execute one step
    gi.step()
    gi.event_dispatcher().process_events()
    
    # Should have received ACTION_EXECUTED
    action_events = [e for e in collector.events 
                     if e.type == dm_ai_module.EventType.ACTION_EXECUTED]
    assert len(action_events) > 0
```

### 期待される効果

- **疎結合**: C++エンジンとPython GUIが独立
- **拡張性**: 複数のUIが同時購読可能（観戦機能など）
- **デバッグ**: イベントログで処理フローを追跡可能

---

## Phase 4: 自動進行システムのC++化（Priority 4）

### 目的
現在PyQt6のQTimerで実装されている自動進行機能を、C++のスレッドベースに移行。

### 現状分析

**ファイル**: [dm_toolkit/gui/app.py](dm_toolkit/gui/app.py#L84-L90)

```python
# Simulation Timer
self.timer = QTimer()
self.timer.timeout.connect(self.session.step_phase)
self.is_running: bool = False

# Auto-start timer for AI vs AI games
if all(mode == 'AI' for mode in self.session.player_modes.values()):
    self.is_running = True
    self.timer.start(500)  # 500ms interval
```

### 問題点

1. **Python GIL**: GILによりスレッド並列性が制限される
2. **タイミング精度**: QTimerは正確ではない（~±50ms）
3. **UI依存**: GUI起動時のみ自動進行可能（CLIでは不可）

### 実装タスク

#### タスク4.1: C++自動進行システム

**新規ファイル**: `src/engine/auto_stepper.hpp`

```cpp
#pragma once
#include "game_instance.hpp"
#include <thread>
#include <atomic>
#include <chrono>

namespace dm::engine {

/**
 * Automatic game stepping in background thread
 */
class AutoStepper {
public:
    explicit AutoStepper(std::shared_ptr<GameInstance> instance);
    ~AutoStepper();

    /**
     * Start automatic stepping
     * @param interval_ms Interval between steps in milliseconds
     */
    void start(int interval_ms = 500);

    /**
     * Stop automatic stepping
     */
    void stop();

    /**
     * Check if currently running
     */
    bool is_running() const { return running_.load(); }

private:
    void step_loop();

    std::shared_ptr<GameInstance> instance_;
    std::thread step_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> interval_ms_{500};
};

} // namespace dm::engine
```

**新規ファイル**: `src/engine/auto_stepper.cpp`

```cpp
#include "auto_stepper.hpp"
#include <iostream>

namespace dm::engine {

AutoStepper::AutoStepper(std::shared_ptr<GameInstance> instance)
    : instance_(instance) {}

AutoStepper::~AutoStepper() {
    stop();
}

void AutoStepper::start(int interval_ms) {
    if (running_.load()) {
        std::cout << "[AutoStepper] Already running\n";
        return;
    }

    interval_ms_.store(interval_ms);
    running_.store(true);
    
    step_thread_ = std::thread(&AutoStepper::step_loop, this);
    std::cout << "[AutoStepper] Started with interval " << interval_ms << "ms\n";
}

void AutoStepper::stop() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    
    if (step_thread_.joinable()) {
        step_thread_.join();
    }
    
    std::cout << "[AutoStepper] Stopped\n";
}

void AutoStepper::step_loop() {
    while (running_.load()) {
        // Check game over
        if (instance_->state.game_over) {
            std::cout << "[AutoStepper] Game over, stopping\n";
            running_.store(false);
            break;
        }

        // Check if current player is human
        PlayerID active = instance_->state.active_player_id;
        if (instance_->state.is_human_player(active)) {
            // Wait for human input
            std::this_thread::sleep_for(
                std::chrono::milliseconds(interval_ms_.load())
            );
            continue;
        }

        // Execute one step
        try {
            bool success = instance_->step();
            if (!success) {
                std::cout << "[AutoStepper] step() returned false\n";
            }
            
            // Process events
            instance_->event_dispatcher()->process_events();
            
        } catch (const std::exception& e) {
            std::cerr << "[AutoStepper] Exception: " << e.what() << "\n";
        }

        // Sleep for interval
        std::this_thread::sleep_for(
            std::chrono::milliseconds(interval_ms_.load())
        );
    }
}

} // namespace dm::engine
```

#### タスク4.2: GameInstanceへの統合

**ファイル**: [src/engine/game_instance.hpp](src/engine/game_instance.hpp)

```cpp
#include "auto_stepper.hpp"

class GameInstance {
public:
    // ... existing ...
    
    // Auto-stepping
    void start_auto_step(int interval_ms = 500) {
        if (!auto_stepper_) {
            auto_stepper_ = std::make_shared<AutoStepper>(
                shared_from_this()  // Requires enable_shared_from_this
            );
        }
        auto_stepper_->start(interval_ms);
    }
    
    void stop_auto_step() {
        if (auto_stepper_) {
            auto_stepper_->stop();
        }
    }
    
    bool is_auto_stepping() const {
        return auto_stepper_ && auto_stepper_->is_running();
    }

private:
    std::shared_ptr<AutoStepper> auto_stepper_;
};
```

**Note**: `shared_from_this()`を使うため、GameInstanceを`enable_shared_from_this`継承に変更:

```cpp
class GameInstance : public std::enable_shared_from_this<GameInstance> {
    // ...
};
```

#### タスク4.3: PyBind11バインディング

**ファイル**: [src/python_bindings/core_bindings.cpp](src/python_bindings/core_bindings.cpp)

```cpp
py::class_<GameInstance, std::shared_ptr<GameInstance>>(m, "GameInstance")
    // ... existing bindings ...
    .def("start_auto_step", &GameInstance::start_auto_step, 
         py::arg("interval_ms") = 500)
    .def("stop_auto_step", &GameInstance::stop_auto_step)
    .def("is_auto_stepping", &GameInstance::is_auto_stepping);
```

#### タスク4.4: Python側の移行

**ファイル**: [dm_toolkit/gui/app.py](dm_toolkit/gui/app.py)

```python
class GameWindow(QMainWindow):
    def __init__(self):
        # ... existing initialization ...
        
        # DELETE: self.timer = QTimer()
        # DELETE: self.timer.timeout.connect(...)
        
        # NEW: Use C++ auto-stepper
        # (No timer needed - C++ handles it)
        
        # Auto-start for AI vs AI
        if all(mode == dm_ai_module.PlayerMode.AI 
               for mode in self.session.gs.player_modes):
            self.session.game_instance.start_auto_step(500)
    
    def on_start_stop_button_clicked(self):
        """Toggle auto-stepping"""
        if self.session.game_instance.is_auto_stepping():
            self.session.game_instance.stop_auto_step()
            self.start_stop_button.setText("開始")
        else:
            self.session.game_instance.start_auto_step(500)
            self.start_stop_button.setText("停止")
```

**ファイル**: [dm_toolkit/gui/game_session.py](dm_toolkit/gui/game_session.py)

```python
# DELETE: toggle_auto_step(), _auto_step_loop(), is_running属性
# These are now in C++ AutoStepper
```

### 注意事項：スレッド安全性

C++スレッドからGameStateを変更する際、Python側でのアクセスと競合する可能性があります。

**対策**:

1. **イベント駆動への完全移行**: Python側はGameStateを直接読まず、イベントのみ受け取る
2. **Mutex保護**: GameState操作時にmutexで保護
3. **GIL解放**: PyBind11で適切にGILを解放

**実装例（PyBind11）**:

```cpp
// Python bindings - release GIL for long operations
.def("step", [](GameInstance& self) {
    py::gil_scoped_release release;  // Release GIL
    bool result = self.step();
    return result;
})
```

### テスト計画

**新規ファイル**: `test_auto_stepper.cpp`

```cpp
TEST(AutoStepper, StartStop) {
    auto db = /* load card db */;
    auto gi = std::make_shared<GameInstance>(42, db);
    
    AutoStepper stepper(gi);
    EXPECT_FALSE(stepper.is_running());
    
    stepper.start(100);  // 100ms interval
    EXPECT_TRUE(stepper.is_running());
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    stepper.stop();
    EXPECT_FALSE(stepper.is_running());
}

TEST(AutoStepper, StopsOnGameOver) {
    auto db = /* load card db */;
    auto gi = std::make_shared<GameInstance>(42, db);
    gi->state.game_over = true;
    
    AutoStepper stepper(gi);
    stepper.start(100);
    
    // Should auto-stop when game is over
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_FALSE(stepper.is_running());
}
```

### 期待される効果

- **性能向上**: GILの影響を受けず、より高速な進行
- **精度向上**: スレッドベースで正確なタイミング制御
- **独立性**: CLI/GUIどちらでも使用可能

---

## Phase 5: レガシーラッパー削除（Long-term）

### 目的
Phase 1-4完了後、不要になったPython側のラッパーコードを削除し、C++直接利用に統一。

### 削除対象ファイル

#### 完全削除候補

1. **[dm_toolkit/commands.py](dm_toolkit/commands.py)** (~1000行)
   - C++のActionを直接使用するため不要

2. **[dm_toolkit/command_builders.py](dm_toolkit/command_builders.py)** (~500行)
   - コマンド構築もC++側で実施

3. **[dm_toolkit/unified_execution.py](dm_toolkit/unified_execution.py)** (~300行)
   - C++ GameInstance.resolve_action()に統一

4. **[dm_toolkit/compat_wrappers.py](dm_toolkit/compat_wrappers.py)** (~200行)
   - 互換性レイヤーが不要に

#### 簡素化候補

1. **[dm_toolkit/engine/compat.py](dm_toolkit/engine/compat.py)**
   - C++バインディングへの薄いラッパーのみ残す
   - 現状の複雑な変換ロジックを削除

### 移行手順

#### ステップ1: 依存関係分析

```powershell
# 各ファイルの使用箇所を調査
Select-String -Path dm_toolkit/**/*.py -Pattern "from dm_toolkit.commands import|import commands"
Select-String -Path dm_toolkit/**/*.py -Pattern "from dm_toolkit.unified_execution import"
# ... 他のファイルも同様
```

#### ステップ2: 段階的削除

1. まず新しいC++直接利用コードを追加
2. 旧コードを`# DEPRECATED`としてマーク
3. 全テストが通ることを確認
4. DEPRECATEDコードを削除

#### ステップ3: テスト更新

全てのテストファイルを更新:

```python
# OLD:
from dm_toolkit.commands import generate_legal_commands
cmds = generate_legal_commands(gs, card_db)

# NEW:
actions = dm_ai_module.IntentGenerator.generate_legal_actions(gs, card_db)
```

### 期待される効果

- **コード削減**: ~2000行のPythonコードを削除
- **保守性向上**: ロジックがC++に一元化
- **性能向上**: 変換オーバーヘッド完全削除

---

## 実装スケジュール

```
Week 1:
  [Mon-Tue] Phase 1: AI選択ロジック統一
  [Wed]     Phase 2: プレイヤーモード管理C++化
  [Thu-Fri] Phase 3: イベント通知システム（設計・実装）

Week 2:
  [Mon]     Phase 3: イベント通知システム（テスト・統合）
  [Tue-Wed] Phase 4: 自動進行スレッド化
  [Thu-Fri] 統合テスト・バグ修正

Week 3:
  [Mon-Wed] Phase 5: レガシーラッパー削除（段階的）
  [Thu]     全体テスト
  [Fri]     ドキュメント更新・レビュー
```

---

## リスク管理

### 高リスク項目

1. **スレッド安全性** (Phase 4)
   - **対策**: Mutex保護、GIL適切な解放、イベント駆動設計

2. **既存機能の破壊** (Phase 5)
   - **対策**: 段階的移行、全テスト実行、DEPRECATED期間設定

3. **性能劣化** (Phase 3)
   - **対策**: イベントキュー最適化、バッチ処理

### 緩和策

- 各Phaseごとに全テスト実行
- レグレッションテスト自動化
- ロールバック手順の文書化

---

## 成功指標

### 定量指標

- [ ] コード削減: Python側 -30% (約2000行削除)
- [ ] ビルド時間: ±5%以内（大幅増加なし）
- [ ] 実行速度: +10-30%向上（GIL除去効果）
- [ ] テストカバレッジ: 80%以上維持

### 定性指標

- [ ] コード可読性向上（レビュアー評価）
- [ ] 拡張性向上（新機能追加が容易）
- [ ] デバッグ容易性向上（イベントログ活用）

---

**作成日**: 2026年2月7日  
**関連ドキュメント**: [GAME_STARTUP_FLOW_ANALYSIS.md](GAME_STARTUP_FLOW_ANALYSIS.md)
