# アプリケーションフロー - クイックリファレンス

**最終更新**: 2026年2月9日

---

## 🚀 起動からシミュレーション開始までの最短フロー

```
1. scripts/run_gui.ps1
   ↓
2. dm_toolkit/gui/app.py :: main()
   └─ GameWindow.__init__()
   ↓
3. dm_toolkit/gui/game_session.py :: __init__()
   ↓
4. GameSession.initialize_game()
   ↓
   4.1 JsonLoader.load_cards("data/cards.json")     [C++]
   4.2 GameInstance(seed, card_db)                  [C++]
   4.3 GameState.setup_test_duel()                  [C++/Python]
   4.4 GameState.set_deck(0, deck)                  [C++]
   4.5 GameState.set_deck(1, deck)                  [C++]
   4.6 PhaseManager.start_game()                    [C++]
   4.7 PhaseManager.fast_forward()                  [C++]
   ↓
5. LayoutBuilder.build() (UI描画開始)
   ↓
6. タイマー開始 (AI vs AI の場合)
   ↓
7. ゲームループ: GameSession.step_game() (毎500ms)
```

---

## 📍 「どこを修正したい？」に対する該当ファイル

| 目的 | 担当ファイル | 関数/クラス |
|------|-----------|-----------|
| **UI 見た目を変更したい** | layout_builder.py | LayoutBuilder.build() |
| **ウィンドウレイアウト変更** | app.py | GameWindow.__init__() |
| **ログ表示を変更** | log_viewer.py | LogViewer.log_message() |
| **ユーザー入力処理** | input_handler.py | GameInputHandler |
| **ゲーム開始処理を変更** | game_session.py | initialize_game() |
| **デッキ設定ロジック** | game_session.py | DEFAULT_DECK |
| **合法アクション生成** | commands.py | generate_legal_commands() |
| **ゲーム進行ロジック（C++）** | game_instance.cpp | step()、resolve_action() |
| **フェーズ遷移ロジック** | phase_manager.cpp | start_game()、next_phase() |
| **カード効果実装** | game_logic_system.cpp | resolve_action() |
| **カードデータベース** | data/cards.json | JSON定義 |

---

## 🔄 重要なメソッド呼び出し順序

### ゲーム初期化シーケンス
```
GameWindow.__init__()
  ↓
GameSession.__init__()
  ↓
GameWindow.initialize_game()
  ↓
GameSession.initialize_game()
  ├─ JsonLoader.load_cards()
  ├─ GameInstance()
  ├─ GameState.setup_test_duel()
  ├─ GameState.set_deck()
  ├─ PhaseManager.start_game()
  └─ PhaseManager.fast_forward()
```

### ゲーム実行ループシーケンス（AI vs AI）
```
timer.timeout (毎500ms)
  ↓
GameSession.step_game()
  ↓
GameInstance.step() [C++]
  ├─ IntentGenerator.generate_legal_actions()
  ├─ AI selector
  └─ GameInstance.resolve_action()
  ↓
GameState 更新
  ↓
callback_update_ui()
  ↓
LayoutBuilder.update_player_display()
```

### ユーザーアクション実行シーケンス
```
GameInputHandler.mouseClick()
  ↓
generate_legal_commands()
  ↓
GameSession.execute_action()
  ↓
ensure_executable_command()
  ↓
GameInstance.resolve_action() [C++]
  ↓
GameState 更新
  ↓
callback_update_ui()
```

---

## 💾 重要な定数・設定値

| 項目 | 値 | 定義ファイル |
|------|-----|-----------|
| **デフォルトデッキサイズ** | 40 | game_session.py |
| **デフォルトデッキ内容** | [1-10]×4 | game_session.py (DEFAULT_DECK) |
| **初期シールド** | 5枚 | phase_manager.cpp |
| **初期手札** | 5枚 | phase_manager.cpp |
| **ゲームループ周期** | 500ms | app.py |
| **初期ターン番号** | 1 | game_state.cpp |
| **初期アクティブプレイ** | 0 (Player 0) | game_state.cpp |
| **シード値** | 42 | game_session.py (デフォルト) |

---

## 🔗 Python-C++ インターフェース

### 提供される C++ 関数（Python から呼び出し可）

```python
# CardDatabase
dm_ai_module.JsonLoader.load_cards(path) -> CardDatabase

# GameInstance
gi = dm_ai_module.GameInstance(seed, card_db)
gi.step() -> bool
gi.resolve_action(action)
gi.initialize_card_stats(deck_size)

# PhaseManager
dm_ai_module.PhaseManager.start_game(gs, card_db)
dm_ai_module.PhaseManager.fast_forward(gs, card_db)
dm_ai_module.PhaseManager.next_phase(gs, card_db)

# GameState (アクセサ)
gs.setup_test_duel()
gs.set_deck(player_id, deck_ids)
gs.is_human_player(player_id) -> bool
gs.clone() -> GameState
```

---

## 📊 ゲーム状態の主要属性

```python
# GameState
gs.turn_number: int              # 現在ターン（1始まり）
gs.active_player_id: int         # 現在プレイヤー（0 or 1）
gs.current_phase: Phase          # 現在フェーズ（MANA, MAIN, ATTACK, END）
gs.game_over: bool               # ゲーム終了フラグ
gs.winner: int                   # 勝者ID（-1 = 未決定）

# Player ゾーン
gs.players[0].hand              # 手札リスト
gs.players[0].mana_zone         # マナゾーン
gs.players[0].battle_zone       # バトルゾーン
gs.players[0].shield_zone       # シールドゾーン
gs.players[0].graveyard         # 墓地
gs.players[0].deck              # デッキ

# プレイヤーモード
gs.player_modes[0]              # PlayerMode (AI or HUMAN)
gs.is_human_player(0) -> bool   # Human プレイヤー判定
```

---

## 🎯 よくある修正シーン

### シーン 1: デッキの初期カード配置を変更
```
対象ファイル: game_session.py
修正箇所: DEFAULT_DECK = [1,2,3,4,5,6,7,8,9,10]*4
```

### シーン 2: UI レイアウト全体変更
```
対象ファイル: layout_builder.py
修正箇所: LayoutBuilder.build()
手階段: プレイヤー表示、ゾーン表示の QWidget 配置
```

### シーン 3: フェーズの進み方を変更
```
対象ファイル: src/engine/systems/flow/phase_manager.cpp
修正箇所: PhaseManager::next_phase()
```

### シーン 4: 初期手札が5枚→3枚に変更したい
```
対象ファイル: src/engine/systems/flow/phase_manager.cpp
修正箇所: PhaseManager::start_game() 内のループ回数 (5→3)
```

### シーン 5: アクション生成ロジック変更
```
対象ファイル: src/engine/systems/intent/intent_generator.cpp
修正箇所: IntentGenerator::generate_legal_actions()
```

---

## 📋 デバッグ・テスト時のチェックポイント

| チェック項目 | 確認ファイル | 期待値 |
|-----------|-----------|--------|
| ゲーム起動成功か | app.py | ウィンドウ表示 |
| GameSession 初期化成功か | game_session.py | 例外なし |
| GameInstance 作成成功か | C++ binding | 有効なポインタ |
| デッキ設定成功か | game_session.py | 40 cards/player |
| ゲーム開始後の状態 | game_state (gs) | Phase=MAIN, Turn=1 |
| アクション生成か | intent_generator.cpp | 合法アクションリスト |
| UI 更新されたか | layout_builder.py | 画面表示変化 |

---

## 🔍 トレース用ログポイント（デバッグ用）

```python
# game_session.py :: initialize_game()
print("1. JsonLoader.load_cards() - 開始")
print("2. GameInstance() - 開始")
print("3. GameState.setup_test_duel() - 開始")
print("4. GameState.set_deck() - 開始")
print("5. PhaseManager.start_game() - 開始")
print("6. PhaseManager.fast_forward() - 開始")
print(f"   Turn={gs.turn_number}, Phase={gs.current_phase}, P0 Hand={len(gs.players[0].hand)}")

# game_session.py :: step_game()
print(f"step_game() - Turn={gs.turn_number}, Active={gs.active_player_id}, Phase={gs.current_phase}")
print(f"  P0: Hand={len(gs.players[0].hand)}, Deck={len(gs.players[0].deck)}")

# commands.py :: generate_legal_commands()
print(f"generate_legal_commands() - Phase={gs.current_phase}, Generated={len(cmds)} commands")
```

---

## ✅ チェックリスト

### アプリ起動時に実行される処理
- [ ] scripts/run_gui.ps1 実行
- [ ] dm_toolkit/gui/app.py ロード
- [ ] GameWindow.__init__() 実行
- [ ] GameSession.__init__() 実行
- [ ] JsonLoader.load_cards() 実行 ← **C++ 初接触**
- [ ] GameInstance() 作成 ← **ゲームエンジン初期化**
- [ ] ゲーム開始フェーズ実行
- [ ] UI 描画開始
- [ ] タイマー開始 (AI vs AI の場合)

### ゲーム実行ループ
- [ ] タイマーが 500ms ごとに step_game() 呼び出し
- [ ] GameInstance.step() が実行される
- [ ] IntentGenerator がアクション候補生成
- [ ] Air selector が最初のアクション選択
- [ ] GameInstance.resolve_action() が実行
- [ ] GameState が更新
- [ ] UI callback が実行
- [ ] LayoutBuilder が表示更新

---

## 📞 重要な連絡先（ファイル）

| 問題 | 確認ファイル |
|------|-----------|
| GUI が起動しない | app.py, layout_builder.py |
| ゲーム初期化失敗 | game_session.py, C++ bindings |
| フェーズが進まない | phase_manager.cpp |
| キャラクターが動かない | intent_generator.cpp, commands.py |
| UI が更新されない | callback_update_ui() → layout_builder.py |
| デッキが消える | game_state.cpp, json_loader.cpp |
| AI が動かない | game_instance.cpp :: step() |

---

## 🎓 新規ファイル追加時のチェック

新しいファイルを追加する際は：

1. **Python ファイルの場合**
   - dm_toolkit/gui/ に追加？ → app.py から import
   - dm_toolkit/ に追加？ → 依存関係確認
   - テストファイル？ → tests/ に追加

2. **C++ ファイル の場合**
   - src/core/ に追加？ → GameState 関連か
   - src/engine/ に追加？ → ゲームロジック関連か
   - src/bindings/ に追加？ → Python バインディング追加必須

3. **データファイルの場合**
   - data/ に追加？ → JSON バリデーション必須
   - data/scenarios/ に追加？ → フォーマット確認必須

---

## 📞 このドキュメント群

1. **APPLICATION_FLOW_AND_FILES_MAPPING.md** ← 全体フロー図
2. **DETAILED_FILE_MAPPING.md** ← ファイル詳細説明
3. **このファイル** ← クイックリファレンス ← **ここから始める！**

**推奨**: 最初に「クイックリファレンス」で全体把握 → 詳細は他の2つを参照
