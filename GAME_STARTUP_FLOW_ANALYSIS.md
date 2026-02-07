# ゲーム開始処理フロー分析レポート

## 概要
本レポートでは、DM_simulationプロジェクトにおけるゲーム開始から進行までの処理フローを、担当ファイルと責務の観点から詳細に分析します。

---

## 1. ゲーム開始処理フロー

### 1.1 エントリーポイント
**ファイル**: [scripts/run_gui.ps1](scripts/run_gui.ps1)
- **処理**: GUI起動スクリプト
- **実行内容**: `pythonw dm_toolkit/gui/app.py` を呼び出し

### 1.2 GUIアプリケーション初期化
**ファイル**: [dm_toolkit/gui/app.py](dm_toolkit/gui/app.py)

#### 処理順序:
```python
1. GameWindow.__init__()
   ├─ LogViewer作成 (ログUI)
   ├─ GameSession作成 (ゲームロジック管理)
   ├─ GameInputHandler作成 (入力処理)
   ├─ カードDB読み込み (EngineCompat.load_cards_robust)
   ├─ session.initialize_game() 呼び出し
   ├─ LayoutBuilder.build() (UI構築)
   └─ デフォルトデッキ自動配備
```

**主要責務**:
- UI初期化とレイアウト構築
- ゲームセッションとの連携
- カードデータベース管理
- タイマー制御 (AI vs AI時の自動進行)

---

### 1.3 ゲームセッション初期化
**ファイル**: [dm_toolkit/gui/game_session.py](dm_toolkit/gui/game_session.py)

#### `initialize_game()` 処理詳細:

```python
def initialize_game(self, card_db, seed=42):
    # 1. C++ CardDatabaseロード
    self.native_card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    
    # 2. GameInstance作成 (C++オブジェクト)
    self.game_instance = dm_ai_module.GameInstance(seed, self.native_card_db)
    self.gs = self.game_instance.state  # GameStateへのエイリアス
    
    # 3. デュエル基本設定
    self.gs.setup_test_duel()
    
    # 4. デッキ設定
    self.gs.set_deck(0, deck0)
    self.gs.set_deck(1, deck1)
    
    # 5. ゲーム開始 (C++ PhaseManager)
    dm_ai_module.PhaseManager.start_game(self.gs, self.native_card_db)
    dm_ai_module.PhaseManager.fast_forward(self.gs, self.native_card_db)
```

**主要責務**:
- C++エンジンとの接続
- ゲームインスタンス管理
- デッキ設定
- ゲーム進行管理（C++への委譲）

**依存するC++モジュール**:
- `dm_ai_module.GameInstance`
- `dm_ai_module.JsonLoader`
- `dm_ai_module.PhaseManager`

---

### 1.4 C++ゲームインスタンス初期化
**ファイル**: [src/engine/game_instance.cpp](src/engine/game_instance.cpp)

#### `GameInstance::GameInstance()` 処理:

```cpp
GameInstance::GameInstance(uint32_t seed, shared_ptr<CardDB> db)
    : state(seed), card_db(db) {
    initial_seed_ = seed;
    
    // トリガー管理システム初期化
    trigger_manager = make_shared<TriggerManager>();
    
    // パイプライン実行システム初期化
    pipeline = make_shared<PipelineExecutor>();
    
    // イベント処理セットアップ
    TriggerManager::setup_event_handling(state, trigger_manager, card_db);
}
```

**主要責務**:
- ゲーム状態の保持
- トリガー管理システムの構築
- パイプライン実行システムの構築
- イベント処理基盤の構築

---

### 1.5 ゲーム開始処理（C++ PhaseManager）
**ファイル**: [src/engine/systems/flow/phase_manager.cpp](src/engine/systems/flow/phase_manager.cpp)

#### `PhaseManager::start_game()` 処理:

```cpp
void PhaseManager::start_game(GameState& game_state, const CardDB& card_db) {
    game_state.turn_number = 1;
    game_state.active_player_id = 0;
    
    // 両プレイヤーに対して:
    for (auto& player : game_state.players) {
        // 1. シールド5枚配置
        for (int i = 0; i < 5; ++i) {
            move_card_cmd(player.deck, Zone::DECK, Zone::SHIELD, player.id);
        }
        
        // 2. 初期手札5枚ドロー
        for (int i = 0; i < 5; ++i) {
            move_card_cmd(player.deck, Zone::DECK, Zone::HAND, player.id);
        }
    }
    
    // 3. ターン開始処理呼び出し
    start_turn(game_state, card_db);
    
    // 4. ループチェック状態更新
    game_state.update_loop_check();
}
```

**主要責務**:
- 初期配置（シールドゾーン、手札）
- ターン管理の開始
- ゲーム状態の整合性チェック

---

## 2. ゲーム進行処理フロー

### 2.1 メインループ（Python側）
**ファイル**: [dm_toolkit/gui/game_session.py](dm_toolkit/gui/game_session.py)

#### `step_game()` 処理:

```python
def step_game(self):
    """ゲーム進行メインループ - C++エンジンに完全委譲"""
    
    # 1. ゲームオーバーチェック
    if self.is_game_over():
        return
    
    active_pid = EngineCompat.get_active_player_id(self.gs)
    is_human = (self.player_modes.get(active_pid) == 'Human')
    
    if is_human:
        # 人間プレイヤー: アクション生成 & 待機
        cmds = generate_legal_commands(self.gs, self.card_db)
        if not cmds:
            self._fast_forward()
        return
    
    # AIプレイヤー: C++ step()で完全自動化
    success = self.game_instance.step()
    
    if not success:
        # 失敗時はfast_forwardでフォールバック
        self._fast_forward()
    
    self.callback_update_ui()
```

**主要責務**:
- ゲーム進行タイミング制御
- 人間/AIの分岐処理
- UI更新トリガー
- C++エンジンへの委譲

---

### 2.2 C++ゲーム進行処理
**ファイル**: [src/engine/game_instance.cpp](src/engine/game_instance.cpp)

#### `GameInstance::step()` 処理:

```cpp
bool GameInstance::step() {
    // 1. ゲームオーバーチェック
    if (state.game_over) return false;
    
    // 2. 合法アクション生成
    auto actions = IntentGenerator::generate_legal_actions(state, *card_db);
    
    if (actions.empty()) {
        // アクションなし → fast_forwardで次の決定点まで進める
        PhaseManager::fast_forward(state, *card_db);
        
        // 再生成して確認
        actions = IntentGenerator::generate_legal_actions(state, *card_db);
        if (actions.empty()) return false;  // スタック判定
    }
    
    // 3. AIアクション選択（優先順位あり）
    const Action* selected = nullptr;
    
    // 優先度1: RESOLVE_EFFECT（保留効果の解決）
    // 優先度2: PLAY_CARD（カードプレイ）
    // 優先度3: ATTACK（攻撃）
    // 優先度4: MANA_CHARGE（マナチャージ）
    // 優先度5: その他アクション
    // 優先度6: PASS（フェイズ終了）
    
    // 4. 選択アクション実行
    if (selected) {
        resolve_action(*selected);
        return true;
    }
    
    return false;
}
```

**主要責務**:
- アクション生成の制御
- AI思考（優先順位ベースの選択）
- アクション実行の制御
- 自動進行の判断

---

### 2.3 アクション生成処理
**ファイル**: [src/engine/actions/intent_generator.cpp](src/engine/actions/intent_generator.cpp)

#### `IntentGenerator::generate_legal_actions()` 処理:

```cpp
vector<Action> IntentGenerator::generate_legal_actions(
    const GameState& game_state,
    const CardDB& card_db
) {
    // 1. ユーザー入力待機中のチェック
    if (game_state.waiting_for_user_input) {
        // クエリ応答用アクション生成
        return generate_query_response_actions();
    }
    
    // 2. 保留効果の処理
    if (!game_state.pending_effects.empty()) {
        PendingEffectStrategy pending_strategy;
        return pending_strategy.generate(ctx);
    }
    
    // 3. スタック処理
    StackStrategy stack_strategy;
    auto stack_actions = stack_strategy.generate(ctx);
    if (!stack_actions.empty()) {
        return stack_actions;
    }
    
    // 4. フェイズ別戦略
    switch (game_state.current_phase) {
        case Phase::START_OF_TURN:
        case Phase::DRAW:
            return {};  // 自動進行フェイズ
            
        case Phase::MANA:
            return ManaPhaseStrategy().generate(ctx);
            
        case Phase::MAIN:
            return MainPhaseStrategy().generate(ctx);
            
        case Phase::ATTACK:
            return AttackPhaseStrategy().generate(ctx);
            
        case Phase::BLOCK:
            return BlockPhaseStrategy().generate(ctx);
            
        default:
            return {};
    }
}
```

**主要責務**:
- フェイズ・状態に応じたアクション生成
- 戦略パターンの適用
- 優先順位の決定
- デバッグログ出力

**関連ファイル**:
- [src/engine/actions/strategies/main_phase_strategy.cpp](src/engine/actions/strategies/main_phase_strategy.cpp)
- [src/engine/actions/strategies/pending_strategy.cpp](src/engine/actions/strategies/pending_strategy.cpp)
- [src/engine/actions/strategies/stack_strategy.cpp](src/engine/actions/strategies/stack_strategy.cpp)
- [src/engine/actions/strategies/phase_strategies.cpp](src/engine/actions/strategies/phase_strategies.cpp)

---

### 2.4 アクション実行処理
**ファイル**: [src/engine/game_instance.cpp](src/engine/game_instance.cpp)

#### `GameInstance::resolve_action()` 処理:

```cpp
void GameInstance::resolve_action(const Action& action) {
    // 1. パイプライン設定
    state.active_pipeline = pipeline;
    
    // 2. 重複実行防止（シグネチャチェック）
    uint64_t sig = make_signature(action);
    if (is_duplicate(sig)) return;
    
    // 3. GameLogicSystemへ委譲
    GameLogicSystem::resolve_action(state, action, *card_db, pipeline);
    
    // 4. シグネチャ記録
    record_signature(sig);
}
```

**主要責務**:
- アクション実行の制御
- 重複実行の防止
- パイプライン実行システムへの委譲
- 実行履歴の管理

---

## 3. 各レイヤーの責務整理

### 3.1 Python GUI層（dm_toolkit/gui/）

| ファイル | 主要クラス | 責務 |
|---------|----------|------|
| [app.py](dm_toolkit/gui/app.py) | GameWindow | UI管理、イベント処理、レイアウト構築 |
| [game_session.py](dm_toolkit/gui/game_session.py) | GameSession | ゲーム進行管理、C++エンジンとの橋渡し |
| [input_handler.py](dm_toolkit/gui/input_handler.py) | GameInputHandler | ユーザー入力処理、カードクリック処理 |
| [layout_builder.py](dm_toolkit/gui/layout_builder.py) | LayoutBuilder | UI要素の配置とウィジェット生成 |

**特徴**:
- C++エンジンへの薄いラッパー
- UI更新とイベント処理に専念
- ゲームロジックはC++に委譲

---

### 3.2 Python統合層（dm_toolkit/）

| ファイル | 主要内容 | 責務 |
|---------|---------|------|
| [commands.py](dm_toolkit/commands.py) | generate_legal_commands | C++アクション→Pythonコマンドラッパー生成 |
| [unified_execution.py](dm_toolkit/unified_execution.py) | ensure_executable_command | コマンド正規化とバリデーション |
| [engine/compat.py](dm_toolkit/engine/compat.py) | EngineCompat | C++/Python互換性レイヤー |

**特徴**:
- C++とPythonの型変換
- レガシーコード対応
- 互換性維持

---

### 3.3 C++ エンジンコア（src/engine/）

#### 3.3.1 ゲーム管理層

| ファイル | クラス | 責務 |
|---------|-------|------|
| [game_instance.cpp/hpp](src/engine/game_instance.cpp) | GameInstance | ゲーム全体の統括、進行制御 |
| [systems/flow/phase_manager.cpp](src/engine/systems/flow/phase_manager.cpp) | PhaseManager | フェイズ遷移、ターン管理 |

#### 3.3.2 アクション層

| ファイル | クラス | 責務 |
|---------|-------|------|
| [actions/intent_generator.cpp](src/engine/actions/intent_generator.cpp) | IntentGenerator | 合法アクション生成の統括 |
| [actions/strategies/main_phase_strategy.cpp](src/engine/actions/strategies/main_phase_strategy.cpp) | MainPhaseStrategy | メインフェイズのアクション生成 |
| [actions/strategies/pending_strategy.cpp](src/engine/actions/strategies/pending_strategy.cpp) | PendingEffectStrategy | 保留効果のアクション生成 |
| [actions/strategies/stack_strategy.cpp](src/engine/actions/strategies/stack_strategy.cpp) | StackStrategy | スタック処理のアクション生成 |

#### 3.3.3 システム層

| ファイル | クラス | 責務 |
|---------|-------|------|
| [systems/game_logic_system.cpp](src/engine/systems/game_logic_system.cpp) | GameLogicSystem | アクション実行ロジック |
| [systems/command_system.cpp](src/engine/systems/command_system.cpp) | CommandSystem | コマンド実行エンジン |
| [systems/trigger_system/trigger_manager.cpp](src/engine/systems/trigger_system/trigger_manager.cpp) | TriggerManager | トリガー管理 |
| [systems/pipeline_executor.cpp](src/engine/systems/pipeline_executor.cpp) | PipelineExecutor | 連鎖処理実行 |
| [systems/card/effect_system.cpp](src/engine/systems/card/effect_system.cpp) | EffectSystem | 効果解決システム |
| [systems/mana/mana_system.cpp](src/engine/systems/mana/mana_system.cpp) | ManaSystem | マナ管理 |

---

## 4. データフロー図

```
┌─────────────────────────────────────────────────────────────┐
│                        GUI Layer (Python)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ GameWindow  │→ │ GameSession  │→ │ InputHandler │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
│         ↓                ↓                                   │
│    UI Update      step_game()                               │
└─────────────────────────│────────────────────────────────────┘
                         │
                         ↓ (PyBind11)
┌─────────────────────────────────────────────────────────────┐
│                    C++ Engine Core                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ GameInstance                                         │   │
│  │  ├─ step()                                           │   │
│  │  └─ resolve_action()                                 │   │
│  └────────────┬─────────────────────────┬─────────────────┘ │
│               ↓                         ↓                    │
│  ┌─────────────────────┐   ┌────────────────────────┐       │
│  │ IntentGenerator     │   │ PhaseManager           │       │
│  │  └─ generate()      │   │  ├─ start_game()       │       │
│  └──────────┬──────────┘   │  ├─ start_turn()       │       │
│             ↓              │  └─ fast_forward()     │       │
│  ┌─────────────────────┐   └────────────────────────┘       │
│  │ Strategy Pattern    │                                     │
│  │  ├─ MainPhase       │                                     │
│  │  ├─ PendingEffect   │                                     │
│  │  └─ Stack           │                                     │
│  └──────────┬──────────┘                                     │
│             ↓                                                │
│  ┌──────────────────────────────────────────────────┐       │
│  │ GameLogicSystem                                  │       │
│  │  └─ resolve_action()                             │       │
│  └────────────┬─────────────────────────────────────┘       │
│               ↓                                              │
│  ┌────────────────────────────────────────────┐             │
│  │ Systems (Mana, Effect, Trigger, Pipeline) │             │
│  └────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      GameState                              │
│  (ゲーム状態の完全な表現)                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 責務分割の評価

### 5.1 現状の強み ✅

1. **レイヤー分離が明確**
   - GUI層、統合層、エンジン層が適切に分離
   - 各層の責務が明確

2. **C++への委譲が進んでいる**
   - ゲームロジックの大部分がC++化済み
   - Pythonは主にUI制御のみ

3. **戦略パターンの活用**
   - フェイズごとのアクション生成が整理されている
   - 拡張性が高い設計

4. **イベント駆動アーキテクチャ**
   - トリガーシステムとパイプライン実行が連携
   - 複雑な連鎖処理に対応

---

### 5.2 改善の余地 ⚠️

1. **Python側のラッパー層が厚い**
   - `commands.py`、`unified_execution.py`などのレガシー対応コード
   - →提案: C++側で直接Pythonバインディング最適化

2. **GameSessionの責務が多い**
   - ゲーム進行管理
   - UI更新制御
   - 入力待機管理
   - プレイヤーモード管理
   - →提案: 後述の分割案参照

3. **C++とPythonの状態同期**
   - `self.gs = self.game_instance.state`のような明示的な同期が必要
   - →提案: C++側でObserverパターン導入

4. **AI選択ロジックの二重実装**
   - Python側（GameSession._select_ai_action）
   - C++側（GameInstance::step）
   - に同様のロジック
   - →提案: 完全にC++に統一

---

## 6. 責務分割改善案

### 6.1 Python側の再編成

#### 提案1: GameSessionの分割

**現状**: GameSessionが多くの役割を持つ

**改善案**: 3つのクラスに分割

```python
# dm_toolkit/gui/game_controller.py
class GameController:
    """ゲーム進行制御専用（C++エンジンとの橋渡し）"""
    def step_game(self): ...
    def execute_action(self, action): ...
    def reset_game(self): ...

# dm_toolkit/gui/player_manager.py
class PlayerManager:
    """プレイヤーモード管理専用"""
    def set_player_mode(self, pid, mode): ...
    def is_human_turn(self): ...
    def get_active_player_mode(self): ...

# dm_toolkit/gui/ui_coordinator.py
class UICoordinator:
    """UI更新とコールバック管理専用"""
    def update_ui(self): ...
    def log_message(self, msg): ...
    def request_input(self): ...
```

**メリット**:
- 単一責任原則の遵守
- テストしやすさの向上
- 再利用性の向上

---

#### 提案2: コマンドラッパーの廃止

**現状**: C++アクション→Pythonコマンドラッパー→C++実行という迂回路

**改善案**: C++アクションを直接PyBind11でバインド

```cpp
// Python binding
py::class_<Action>(m, "Action")
    .def_readonly("type", &Action::type)
    .def_readonly("card_id", &Action::card_id)
    .def_readonly("source_instance_id", &Action::source_instance_id)
    // ... 必要なフィールドを全て公開

// Python側では直接使用
action = Action()
action.type = PlayerIntent.PLAY_CARD
game_instance.resolve_action(action)
```

**メリット**:
- `commands.py`、`command_builders.py`などが不要に
- 変換オーバーヘッド削減
- 型安全性の向上

---

### 6.2 C++側の再編成

#### 提案3: GameInstanceの責務整理

**現状**: GameInstanceがstep()内でAI選択も実行も担当

**改善案**: AI選択ロジックを分離

```cpp
// src/engine/ai/simple_ai.hpp
class SimpleAI {
public:
    static const Action* select_action(
        const std::vector<Action>& actions,
        const GameState& state
    );
};

// game_instance.cpp内
bool GameInstance::step() {
    auto actions = IntentGenerator::generate_legal_actions(state, *card_db);
    if (actions.empty()) { /* ... */ }
    
    // AI選択ロジックを分離
    const Action* selected = SimpleAI::select_action(actions, state);
    
    if (selected) {
        resolve_action(*selected);
        return true;
    }
    return false;
}
```

**メリット**:
- AI実装の切り替えが容易
- より高度なAIの実装が可能
- テストしやすさの向上

---

#### 提案4: PhaseManager の静的メソッドをインスタンス化

**現状**: PhaseManagerが全て静的メソッド

**改善案**: GameInstanceが保持するPhaseManagerインスタンス

```cpp
class GameInstance {
    std::shared_ptr<PhaseManager> phase_manager_;
    
public:
    void start_game() {
        phase_manager_->start_game(state, *card_db);
    }
};

class PhaseManager {
    GameState& state_;
    const CardDB& card_db_;
    
public:
    PhaseManager(GameState& state, const CardDB& card_db)
        : state_(state), card_db_(card_db) {}
    
    void start_game();
    void start_turn();
    void fast_forward();
};
```

**メリット**:
- 状態管理の明確化
- テスト時のモック化が容易
- 拡張性の向上（複数ゲーム並列実行など）

---

## 7. さらなるC++化の提案

### 7.1 完全C++化ターゲット

現在Python側に残っている処理のうち、C++化すべきもの:

#### Priority 1: ゲーム進行タイマー制御

**現状**: [dm_toolkit/gui/app.py](dm_toolkit/gui/app.py)
```python
self.timer = QTimer()
self.timer.timeout.connect(self.session.step_phase)
self.timer.start(500)
```

**提案**: C++側で自動進行スレッド実装

```cpp
// src/engine/game_instance.hpp
class GameInstance {
    std::thread auto_step_thread_;
    std::atomic<bool> auto_stepping_{false};
    
public:
    void start_auto_step(int interval_ms = 500);
    void stop_auto_step();
};
```

**メリット**:
- Python GILの影響を受けない
- より高速な進行が可能
- Pythonはイベント受信のみに専念

---

#### Priority 2: プレイヤーモード管理

**現状**: Python側で管理
```python
self.player_modes: Dict[int, str] = {0: 'AI', 1: 'AI'}
```

**提案**: GameState に統合

```cpp
// core/game_state.hpp
struct GameState {
    enum class PlayerMode { HUMAN, AI };
    std::array<PlayerMode, MAX_PLAYERS> player_modes{PlayerMode::AI, PlayerMode::AI};
};
```

**メリット**:
- ゲーム状態の完全性（全ての情報がC++に）
- 同期不要
- セーブ/ロード対応が容易

---

#### Priority 3: UI更新イベント通知

**現状**: Python側でコールバック呼び出し

**提案**: C++からPythonへのイベント通知システム

```cpp
// src/engine/events/event_dispatcher.hpp
class EventDispatcher {
public:
    using Callback = std::function<void(const GameEvent&)>;
    
    void subscribe(EventType type, Callback callback);
    void emit(const GameEvent& event);
};

// Python binding
game_instance.subscribe(EventType::STATE_CHANGED, lambda e: update_ui())
game_instance.subscribe(EventType::ACTION_EXECUTED, lambda e: log_action(e))
```

**メリット**:
- 疎結合な設計
- 複数のUIが同時に購読可能
- イベント駆動アーキテクチャの実現

---

### 7.2 C++化実装計画

#### フェーズ1: AI選択ロジックの統一（1-2日）

- [ ] SimpleAIクラス実装
- [ ] GameInstance::step()からPython側のロジック削除
- [ ] 動作テスト

#### フェーズ2: プレイヤーモード管理のC++化（1日）

- [ ] GameStateにplayer_modes追加
- [ ] Python側のplayer_modes削除
- [ ] バインディング更新

#### フェーズ3: イベント通知システム構築（2-3日）

- [ ] EventDispatcher実装
- [ ] 主要イベント定義（STATE_CHANGED, ACTION_EXECUTED等）
- [ ] PyBind11でバインディング
- [ ] Python側をコールバックからイベント購読に移行

#### フェーズ4: 自動進行システムのC++化（2-3日）

- [ ] GameInstanceにauto_step機能追加
- [ ] スレッドセーフな実装
- [ ] Python側タイマー削除
- [ ] 性能測定

#### フェーズ5: レガシーラッパー削除（3-5日）

- [ ] commands.py の段階的廃止
- [ ] unified_execution.py の簡素化
- [ ] 直接C++ Action使用への移行
- [ ] 全テストの更新

**総見積もり**: 2-3週間

---

## 8. まとめ

### 現状の評価

**良い点**:
- レイヤー分離が明確
- C++エンジンへの委譲が進んでいる
- 拡張性の高い設計

**改善点**:
- Python側のラッパー層削減の余地あり
- GameSessionの責務が多い
- AI選択ロジックの二重実装

### 推奨手順

1. **短期（1-2週間）**:
   - GameSessionの分割（提案1）
   - AI選択ロジックの統一（フェーズ1）
   - プレイヤーモード管理のC++化（フェーズ2）

2. **中期（1ヶ月）**:
   - イベント通知システム構築（フェーズ3）
   - 自動進行システムのC++化（フェーズ4）

3. **長期（2-3ヶ月）**:
   - レガシーラッパー削除（フェーズ5）
   - PhaseManagerのインスタンス化（提案4）
   - 完全なC++エンジン化

### 期待される効果

- **性能**: GIL除去、変換オーバーヘッド削減により10-30%高速化
- **保守性**: 責務分離により各モジュールが理解しやすく
- **拡張性**: イベント駆動設計により新機能追加が容易に
- **テスト性**: 各コンポーネントの単体テストが可能に

---

**作成日**: 2026年2月7日  
**対象バージョン**: DM_simulation latest (C++ engine with Python GUI)
