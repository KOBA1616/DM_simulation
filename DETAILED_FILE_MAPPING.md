# 担当コンポーネント別ファイル詳細マップ

**作成日**: 2026年2月9日  
**用途**: 開発時の参照、責務分界の確認

---

## 1. GUI層ファイル群

### A. メインウィンドウ層

#### `dm_toolkit/gui/app.py`
```
主要クラス: GameWindow(QMainWindow)
重要メソッド:
  - __init__()              : ウィンドウ初期化
  - update_ui()            : UI全体更新
  - reset_game()           : ゲームリセット
  - closeEvent()           : ウィンドウ閉じる時の処理

重要属性:
  - self.session (GameSession)          : ゲーム管理
  - self.card_db (CardDatabase)         : カード定義
  - self.timer (QTimer)                 : 定期実行タイマー
  - self.is_running (bool)              : ゲーム実行中フラグ
  - self.layout_builder (LayoutBuilder) : UI構築
  - self.log_viewer (LogViewer)         : ログ表示
  - self.input_handler (GameInputHandler) : 入力処理

責務:
  ✓ ウィンドウ・UI初期化
  ✓ ゲームセッション管理
  ✓ ゲームループ制御（timer）
  ✓ UI更新の調整（callback 経由）
  ✓ ゲームリセット
  ✓ デッキロードと配備
```

#### `dm_toolkit/gui/game_session.py`
```
主要クラス: GameSession
重要メソッド:
  - __init__()             : セッション初期化
  - initialize_game()      : ゲーム初期化（最重要）
  - reset_game()           : ゲームリセット
  - step_game()            : ゲームの1ステップ実行
  - execute_action()       : ユーザーアクション実行
  - _fast_forward()        : 自動進行
  - is_game_over()         : ゲームオーバー判定

重要属性:
  - self.game_instance (GameInstance)  : C++ ゲーム
  - self.gs (GameState)               : ゲーム状態
  - self.card_db (CardDB)             : カード定義
  - self.native_card_db (CardDB_C++)  : C++ 側カード定義
  - self.DEFAULT_DECK                 : デフォルトデッキ
  - self.player_modes                 : プレイヤーモード

責務:
  ✓ C++ ゲームインスタンスの管理
  ✓ ゲーム状態の同期
  ✓ デッキ設定
  ✓ ゲーム進行ループ制御
  ✓ ユーザーアクション処理
  ✓ 自動進行（AI）
  ✓ コールバック実行
```

### B. 入力処理層

#### `dm_toolkit/gui/input_handler.py`
```
主要クラス: GameInputHandler
重要メソッド:
  - handle_card_click()    : カードクリック処理
  - handle_player_click()  : プレイヤークリック処理
  - get_selected_cards()   : 選択カード取得

責務:
  ✓ マウスイベント処理
  ✓ カード選択管理
  ✓ アクション候補生成
```

### C. UI構築層

#### `dm_toolkit/gui/layout_builder.py`
```
主要クラス: LayoutBuilder
重要メソッド:
  - build()                : UI構築
  - update_player_display() : プレイヤー表示更新
  - update_zones_display() : ゾーン表示更新
  - update_hand_display()  : 手札表示更新

責務:
  ✓ ウィジェット生成
  ✓ レイアウト配置
  ✓ ゲーム状態に基づく表示更新
```

#### `dm_toolkit/gui/log_viewer.py`
```
主要クラス: LogViewer(QTextEdit)
重要メソッド:
  - log_message()          : メッセージ出力

責務:
  ✓ ゲーム進行ログ表示
  ✓ エラーメッセージ表示
  ✓ ゲーム状態通知表示
```

---

## 2. ゲームロジック層（Python）

### `dm_toolkit/commands.py`
```
重要関数:
  - generate_legal_commands(gs, card_db)
  - declare_play_command(player_id, card_id, ...)
  - attack_creature_command(...)
  - pass_command()
  - mana_charge_command(...)

責務:
  ✓ 法的コマンド候補生成
  ✓ コマンド形式の異なるバージョン生成
  ✓ ゲーム状態に基づく妥当性検証
```

### `dm_toolkit/engine/compat.py`
```
主要クラス: EngineCompat
重要メソッド:
  - load_cards_robust()    : カードDB読み込み（エラーハンドリング付き）
  - ExecuteCommand()       : コマンド実行
  - get_active_player_id() : アクティブプレイヤー取得

責務:
  ✓ C++ エンジンとのインターフェース
  ✓ エラーハンドリング
  ✓ コマンド実行の仲介
```

### `dm_toolkit/unified_execution.py`
```
重要関数:
  - ensure_executable_command()

責務:
  ✓ コマンド形式の統一
  ✓ 実行可能な形式への変換
```

### `dm_toolkit/dm_types.py`
```
重要クラス：
  - GameState
  - CardDB
  - Zone
  - Phase
  - ActionType

責務:
  ✓ ゲーム関連の型定義
  ✓ 列挙体定義
```

---

## 3. Python Fallback 実装

### `dm_ai_module.py`
```
重要クラス:
  - Phase(IntEnum)
    - MANA = 2
    - MAIN = 3
    - ATTACK = 4
    - END = 5
  
  - PlayerMode(IntEnum)
    - AI = 0
    - HUMAN = 1
  
  - GameState
    属性:
      - players[Player]
      - current_phase
      - active_player_id
      - turn_number
      - game_over
      - player_modes[]
    メソッド:
      - setup_test_duel()        [★重要]
      - set_deck(player_id, deck_ids)
      - is_human_player(pid)     [★重要]
      - add_card_to_hand()
      - add_card_to_mana()
      - clone()
  
  - GameInstance
    属性:
      - state (GameState)
      - card_db
    メソッド:
      - start_game()
      - step()                   [★重要]
      - resolve_action()         [★重要]
      - execute_action()
      - initialize_card_stats()
  
  - ActionGenerator
    メソッド:
      - generate_legal_actions(state, card_db)

責務:
  ✓ C++ 拡張が利用不可の場合のフォールバック実装
  ✓ テスト用の最小限のゲームロジック提供
  ✓ Python-C++ インターフェースの型定義
```

### `dm_ai_module.pyi`
```
重要定義:
  - Phase, PlayerMode, ActionType の型ヒント
  - GameState, GameInstance の signature
  - ActionGenerator の signature

責務:
  ✓ MyPy による型チェック支援
  ✓ IDE オートコンプリート支援
```

---

## 4. C++ コアエンジン層

### A. ゲーム状態管理

#### `src/core/game_state.cpp/.hpp`
```
主要メソッド:
  - GameState(seed)        : コンストラクタ
  - setup_test_duel()      : テストデュエル初期化
  - set_deck(player_id, deck_ids)
  - is_human_player(player_id)
  - clone()                : ディープコピー
  - get_card_instance(instance_id)
  - get_zone(player_id, zone_type)

重要属性:
  - players[Player]        : プレイヤー配列
  - current_phase          : 現在フェーズ
  - active_player_id       : アクティブプレイヤーID
  - turn_number            : ターン番号
  - game_over              : ゲームオーバーフラグ
  - winner                 : 勝者ID
  - player_modes[]         : プレイヤーモード

責務:
  ✓ ゲーム状態の保持
  ✓ プレイヤー・ゾーン管理
  ✓ フェーズ・ターン管理
```

#### `src/core/types.hpp`
```
主要定義:
  - enum Phase { START, DRAW, MANA, MAIN, ATTACK, END }
  - enum PlayerMode { AI, HUMAN }
  - enum Zone { DECK, HAND, MANA_ZONE, BATTLE, GRAVEYARD, SHIELD }
  - enum CardType { CREATURE, SPELL }
  - using PlayerID = int
  - using CardID = int

責務:
  ✓ 型安全性の確保
  ✓ 列挙値の中央管理
```

### B. ゲーム進行管理

#### `src/engine/game_instance.cpp/.hpp`
```
主要メソッド:
  - GameInstance(seed, card_db)
  - step() -> bool         : アクション実行
  - resolve_action(action) : アクション解決
  - start_game()
  - initialize_card_stats()
  - reset_with_scenario()

重要属性:
  - state (GameState)
  - card_db
  - trigger_manager
  - pipeline

責務:
  ✓ ゲーム進行の主要ループ
  ✓ アクション生成・実行の制御
  ✓ トリガーシステム管理
  ✓ パイプライン実行
```

### C. フェーズ・ターン管理

#### `src/engine/systems/flow/phase_manager.cpp/.hpp`
```
主要メソッド:
  - start_game(gs, card_db)
    処理:
      1. シールド5枚配置
      2. 手札5枚ドロー
      3. start_turn() 呼び出し
  
  - start_turn(gs, card_db)
    処理:
      1. ターン統計リセット
      2. アンタップ処理
      3. ドローフェーズ（P0はスキップ）
  
  - fast_forward(gs, card_db)
    処理:
      1. 合法アクションなくなるまで進行
      2. 次の決定点に到達
  
  - next_phase(gs, card_db)

責務:
  ✓ ゲーム開始時初期化
  ✓ フェーズ遷移制御
  ✓ ターン開始・終了処理
  ✓ 自動進行
```

### D. アクション生成

#### `src/engine/systems/intent/intent_generator.cpp/.hpp`
```
主要メソッド:
  - generate_legal_actions(state, card_db)
    返値: vector<Action>
    
    フェーズ別アクション:
      - MANA: MANA_CHARGE
      - MAIN: DECLARE_PLAY, PASS
      - ATTACK: ATTACK_CREATURE, PASS
      - END: PASS

責務:
  ✓ 現在ゲーム状態から法的アクション候補を生成
  ✓ フェーズに基づくアクション制限
  ✓ プレイヤー状態に基づく候補選別
```

### E. カードデータベース

#### `src/engine/systems/card/json_loader.cpp/.hpp`
```
主要メソッド:
  - JsonLoader::load_cards(path) -> CardDatabase*
    処理:
      1. JSON ファイル読み込み
      2. CardDefinition でパース
      3. CardRegistry に登録

責務:
  ✓ JSON 形式カード定義読み込み
  ✓ パース・バリデーション
  ✓ カード登録簿への登録
```

#### `src/engine/systems/card/card_registry.cpp/.hpp`
```
主要メソッド:
  - load_from_json(json_data)
  - get_all_definitions_ptr()
  - get_card(card_id)

責務:
  ✓ カード定義の一元管理
  ✓ ID からの高速参照
```

---

## 5. データフロー別ファイル群

### ゲーム状態フロー
```
C++ GameState
  ↓ (Python via pybind11)
Python GameSession.gs
  ↓
Python GameWindow.session.gs
  ↓
Python LayoutBuilder (UI 更新用)
```

### コマンド実行フロー
```
commands.py: generate_legal_commands()
  ↓
input_handler.py: handle_card_click()
  ↓
game_session.py: execute_action()
  ↓
unified_execution.py: ensure_executable_command()
  ↓
C++ GameInstance.resolve_action()
  ↓
C++ GameLogicSystem (effect 解決等)
  ↓
C++ GameState (更新)
  ↓
LayoutBuilder: update_ui() (表示更新)
```

### ゲームループフロー
```
GameWindow.timer (毎 500ms)
  ↓
GameSession.step_game()
  ↓
(Human プレイヤーの場合)
  input_handler で待機
  ↓
(AI プレイヤーの場合)
  GameInstance.step()
    - IntentGenerator.generate_legal_actions()
    - AI selector で最初のアクション選択
    - GameInstance.resolve_action()
  ↓
GameState 更新
  ↓
LayoutBuilder: callback_update_ui()
```

---

## 6. ファイルサイズ・複雑度参考

| ファイル | 推定行数 | 複雑度 | 備考 |
|---------|---------|--------|------|
| app.py | 300-400 | 中 | UI 統合点 |
| game_session.py | 400-500 | 中 | ゲーム管理中核 |
| commands.py | 200-300 | 中 | コマンド生成 |
| phase_manager.cpp | 200-300 | 高 | フェーズ遷移複雑 |
| game_instance.cpp | 300-400 | 高 | AI/トリガー制御 |
| intent_generator.cpp | 150-250 | 中 | アクション生成 |
| game_state.cpp | 200-300 | 中 | 状態管理 |

---

## 7. 重要な依存関係

### Python 層内の依存
```
app.py
  ├─ game_session.py
  ├─ layout_builder.py
  ├─ input_handler.py
  ├─ log_viewer.py
  └─ engine/compat.py

game_session.py
  ├─ dm_ai_module (C++ or Python fallback)
  ├─ engine/compat.py
  └─ commands.py

input_handler.py
  ├─ commands.py
  └─ game_session.py
```

### C++ 層内の依存
```
game_instance.cpp
  ├─ game_state.cpp
  ├─ phase_manager.cpp
  ├─ intent_generator.cpp
  ├─ trigger_manager.cpp
  └─ pipeline_executor.cpp

phase_manager.cpp
  ├─ game_state.cpp
  └─ card_registry.cpp

intent_generator.cpp
  ├─ game_state.cpp
  └─ card_registry.cpp
```

### Python ↔ C++ 間の依存
```
dm_ai_module (バインディング)
  ├─ src/bindings/bind_core.cpp
  ├─ src/bindings/bind_game_instance.cpp
  ├─ src/bindings/bind_phase_manager.cpp
  └─ src/bindings/bind_intent_generator.cpp
```

---

## ✅ 全ファイル完全マッピング確認

**Python GUI層**: 7 ファイル
**Python Logic層**: 4 ファイル
**Python Fallback**: 2 ファイル
**C++ Core層**: 15+ ファイル
**データ層**: 3 ファイル

**合計**: 31+ ファイル

**状態**: ✅ 完全マッピング完了
