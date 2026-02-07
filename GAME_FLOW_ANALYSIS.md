# ゲーム開始から実行される処理フローと担当ファイル

## 1. ゲーム初期化フロー

### Python側 (GUI起動)
```
dm_toolkit/gui/app.py
  └─> GameSession.__init__()
       dm_toolkit/gui/game_session.py
```

**責務**: 
- UIコールバック設定
- ゲームセッション初期化
- プレイヤーモード設定 (Human/AI)

### ゲーム開始処理
```
GameSession.initialize_game()
  ├─> JsonLoader.load_cards("data/cards.json")  [C++]
  │    src/engine/systems/card/json_loader.cpp
  │    └─> CardRegistry::load_from_json()
  │         src/engine/systems/card/card_registry.cpp
  │         
  ├─> GameInstance(seed, card_db)  [C++]
  │    src/engine/game_instance.cpp
  │    └─> 内部でPipelineExecutor, TriggerManager初期化
  │    
  ├─> GameState.setup_test_duel()  [C++]
  │    src/core/game_state.cpp
  │    
  ├─> GameState.set_deck(player_id, deck)  [C++]
  │    
  └─> PhaseManager::start_game()  [C++]
       src/engine/systems/flow/phase_manager.cpp
       ├─> シールド配置 (5枚)
       ├─> 手札ドロー (5枚)
       └─> start_turn()
            ├─> turn_stats リセット
            ├─> アンタップ処理
            └─> ドローフェーズ (Turn 1 P0はスキップ)
```

**責務**:
- CardDatabase読み込み (C++)
- GameInstance生成 (C++)
- デュエル初期状態設定 (C++)
- ゲーム開始処理 (C++)

---

## 2. ゲーム進行ループ

### Python側メインループ
```
GameSession.step_game()
  ├─> Human player → 入力待ち
  │    └─> generate_legal_commands() [Python wrapper]
  │         └─> IntentGenerator::generate_legal_actions() [C++]
  │    
  └─> AI player → 自動実行
       └─> GameInstance.step()  [C++]
            src/engine/game_instance.cpp
```

### C++ step()処理
```
GameInstance::step()
  ├─> IntentGenerator::generate_legal_actions()
  │    src/engine/actions/intent_generator.cpp
  │    └─> PhaseStrategy選択
  │         src/engine/actions/strategies/phase_strategies.cpp
  │         ├─> ManaPhaseStrategy
  │         ├─> MainPhaseStrategy
  │         ├─> AttackPhaseStrategy
  │         └─> BlockPhaseStrategy
  │    
  ├─> AI選択 (優先順位)
  │    1. RESOLVE_EFFECT
  │    2. PLAY_CARD
  │    3. ATTACK
  │    4. MANA_CHARGE
  │    5. その他
  │    6. PASS
  │    
  ├─> resolve_action(selected_action)
  │    └─> GameLogicSystem::resolve_action_oneshot()
  │         src/engine/systems/game_logic_system.cpp
  │         └─> 各種Handler呼び出し
  │    
  └─> fast_forward() (アクション0個時)
       └─> PhaseManager::fast_forward()
            └─> next_phase()まで繰り返し
```

**責務**:
- アクション生成 (C++)
- AI選択ロジック (C++)
- アクション実行 (C++)
- 自動進行 (C++)

---

## 3. アクション実行フロー

```
GameInstance::resolve_action()
  └─> GameLogicSystem::resolve_action_oneshot()
       ├─> MANA_CHARGE → ManaChargeHandler
       ├─> PLAY_CARD → PlayCardHandler
       ├─> ATTACK_PLAYER/CREATURE → AttackHandler
       ├─> RESOLVE_EFFECT → PipelineExecutor
       └─> PASS → PhaseManager::next_phase()
```

**担当ファイル**:
- `src/engine/systems/game_logic_system.cpp` - メインディスパッチャ
- `src/engine/systems/card/handlers/*.cpp` - 各種Handler

---

## 4. フェーズ管理

```
PhaseManager::next_phase()
  src/engine/systems/flow/phase_manager.cpp
  
  START_OF_TURN
    ↓
  UNTAP (P1のみ)
    ↓
  MANA
    ↓
  MAIN
    ↓
  ATTACK
    ↓
  BLOCK (攻撃時のみ)
    ↓
  END_OF_TURN
    ↓
  (次プレイヤーのSTART_OF_TURN へ)
```

**責務**:
- フェーズ遷移管理
- ターン開始/終了処理
- 自動進行制御

---

## 5. 現在の問題: card_db読み込みエラー

### エラー内容
```
Error loading card JSON: [json.exception.type_error.306] cannot use value() with string
Loaded card database: 0 cards
```

### 発生箇所
`src/engine/systems/card/card_registry.cpp` Line 83

### 原因調査中
- JSON構造は正しい（Pythonで確認済み）
- C++パース処理に問題の可能性
- nlohmann::json の使い方の問題

---

## 6. 責務分割の現状評価

### ✅ 適切に分離されている部分
1. **フェーズ管理**: PhaseManager (C++)
2. **アクション生成**: IntentGenerator + PhaseStrategy (C++)
3. **アクション実行**: GameLogicSystem + Handlers (C++)
4. **UI管理**: GameSession (Python)

### ⚠️ 改善が必要な部分
1. **CardDatabase読み込み**: 
   - 現状: JsonLoader (C++) → エラー発生中
   - 提案: エラー原因特定と修正

2. **AI選択ロジック**:
   - 現状: GameInstance::step() (C++)
   - 問題: 単純な優先順位のみ
   - 提案: MCTS/BeamSearchへの統合

3. **Human入力処理**:
   - 現状: Python側でgenerate_legal_commands()
   - 提案: C++のIntentGeneratorを直接使用

---

## 7. C++化の推奨手順

### Phase 1: card_db問題の解決 (最優先)
1. CardRegistry::load_from_json() のデバッグログ追加
2. エラー原因特定
3. 修正実装
4. テスト確認

### Phase 2: GameSession処理のC++化
1. `GameSession.step_game()`の完全C++移行
2. Human入力もIntentGeneratorで統一
3. Pythonは純粋にUIコールバックのみ

### Phase 3: AI統合
1. MCTS/BeamSearchをGameInstance統合
2. AIモード選択機能
3. パフォーマンス最適化
