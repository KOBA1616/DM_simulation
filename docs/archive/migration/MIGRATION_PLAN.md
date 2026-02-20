# Python → C++ ゲーム進行システム段階的移行計画

## 現在の状況（2026/02/05）

### 完了した作業
- ✅ **Step 0**: Python側重複フラグ削除（`_mana_charged_by_state`）
- ✅ **Step 1**: FlowCommand実装完成（SET_MANA_CHARGED, RESET_TURN_STATS, CLEANUP_STEP）
- 🔄 **Step 1.5**: マナチャージフラグのプレイヤーごと管理化（進行中）

### 発見された問題
**問題**: `turn_stats.mana_charged_this_turn`がゲーム全体で1つの値しかない
- **症状**: 同一ターン内でプレイヤー切り替え時にフラグがリセットされ、両プレイヤーがマナチャージ可能
- **解決策**: `bool mana_charged_by_player[2]`に変更（プレイヤーごとに管理）

### 修正内容
1. `src/core/card_stats.hpp`: `TurnStats`構造体のフラグを配列化
2. `src/engine/actions/strategies/phase_strategies.cpp`: `mana_charged_by_player[active_player_id]`参照
3. `src/engine/game_command/commands.cpp`: FlowCommand実装でプレイヤーIDインデックス使用
4. `src/core/game_state_command.cpp`: 未使用ファイルも一貫性のため更新
5. **残作業**: `src/engine/systems/game_logic_system.cpp:1069`の古いコード削除

---

## 段階的実装計画

### **Phase 1: 基盤整備**（現在のフェーズ）

#### Step 1.5: マナチャージフラグ修正【進行中】
- **目的**: プレイヤーごとのマナチャージ制限を正しく実装
- **作業**:
  - [x] `mana_charged_by_player[2]`への変更
  - [ ] ビルド成功
  - [ ] 動作確認（各プレイヤー1ターン1回のみ）
- **検証基準**:
  - ✅ P0がマナチャージ後、同じターンでP1に切り替わってもP0はチャージ不可
  - ✅ P1がマナチャージ可能
  - ✅ 次ターンで両プレイヤーのフラグがリセットされる

#### Step 2: ターン統計の完全移行
- **目的**: `TurnStats`の全フィールドをプレイヤーごとに管理
- **現状**: 以下はゲーム全体で1つ
  ```cpp
  int played_without_mana = 0;
  int cards_drawn_this_turn = 0;
  int cards_discarded_this_turn = 0;
  int creatures_played_this_turn = 0;
  int spells_cast_this_turn = 0;
  int current_chain_depth = 0;
  ```
- **検討事項**: 
  - これらもプレイヤーごとに分ける必要があるか？
  - DMのルールではプレイヤーごとか、ターンごとか？
- **提案**: `TurnStats per_player_stats[2]`に全体をリファクタリング

---

### **Phase 2: フェーズ管理のC++移行**

#### Step 3: PhaseManager完全移行
- **目的**: Python側の`_transition_phase()`を完全にC++化
- **現状**:
  - C++: `PhaseManager::start_turn()`, `advance_phase()`は実装済み
  - Python: `dm_toolkit/commands.py`が依然としてフェーズ遷移を制御
- **作業**:
  1. Python側フェーズ遷移ロジックの棚卸し
  2. C++ `PhaseManager`にロジック移植
  3. Python側は`PhaseManager`への委譲のみに変更
- **検証**: Pythonコードからフェーズ制御コードが完全に消える

#### Step 4: ターンループのC++化
- **目的**: `game_session.py`の`_execute_turn()`をC++に移行
- **現状**:
  - Python: ターンループ、プレイヤー切り替え、勝敗判定を管理
  - C++: 個別コマンド実行のみ
- **作業**:
  1. `GameController`クラスの再設計（Step 1で挫折した経緯あり）
  2. ターンループロジックの移植
  3. 勝敗判定のC++化
- **課題**: ビルドシステムの複雑さ、ヘッダー依存関係

---

### **Phase 3: AI統合とゲームループ**

#### Step 5: AI選択ロジックのC++化
- **目的**: `_ai_select_action()`をC++に移行
- **現状**:
  - Python: AIがアクション選択
  - C++: `ActionGenerator`が合法手生成のみ
- **作業**:
  1. `AIController`クラス作成
  2. MCTSエンジンとの統合
  3. Python側はUI用のラッパーのみに
- **検証**: Python側のAIロジックコードが消える

#### Step 6: 完全自動プレイループ
- **目的**: C++だけでゲーム開始〜終了まで実行
- **作業**:
  1. `GameController::run_game()`実装
  2. デッキ初期化のC++化
  3. 自動プレイモード実装
- **成果物**: `dm_ai_module.auto_play(deck1, deck2)` → 結果返却

---

### **Phase 4: パフォーマンス最適化**

#### Step 7: バッチシミュレーション
- **目的**: 1000ゲーム並列実行を高速化
- **作業**:
  1. GIL解放（`Py_BEGIN_ALLOW_THREADS`）
  2. マルチスレッド対応
  3. ゲーム状態のメモリプール
- **目標**: 10,000ゲーム/秒の処理速度

#### Step 8: ログ・デバッグ機能
- **目的**: C++版デバッグ支援
- **作業**:
  1. 構造化ログ出力
  2. リプレイ機能
  3. プロファイリング統合

---

## 優先順位と推奨アプローチ

### 即座に対応すべき作業
1. **マナチャージフラグ修正の完了**（Step 1.5）
   - 手動でファイル編集 → ビルド → テスト
   - 所要時間: 10分

### 短期目標（1週間）
2. **TurnStats設計の見直し**（Step 2）
   - プレイヤーごとの統計が必要か仕様確認
   - 必要なら`per_player_stats[2]`に変更

3. **PhaseManager移行**（Step 3）
   - Python依存度を大幅に下げる
   - 影響範囲が明確で低リスク

### 中期目標（1ヶ月）
4. **GameController再挑戦**（Step 4）
   - 前回の失敗を踏まえて慎重に設計
   - まず最小実装でビルド成功を優先

### 長期目標（3ヶ月）
5. **完全C++化**（Step 5-8）
   - AI統合、自動プレイ、最適化

---

## リスク管理

### 高リスク作業
- **GameController作成**: 前回失敗した経験あり
  - 対策: 最小実装から始める、段階的に機能追加
- **ビルドシステム**: ヘッダー依存関係が複雑
  - 対策: 前方宣言の活用、循環依存の回避

### 低リスク作業
- **PhaseManager移行**: 既存C++コードの拡張のみ
- **ログ機能追加**: 既存システムへの影響なし

---

## 次のアクション

### 今すぐ実行
1. ✅ VS Codeダイアログで「再試行」クリック
2. ✅ `game_logic_system.cpp:1069`のコード削除
3. ✅ ビルド成功確認
4. ✅ GUIで動作テスト（3ターン以上プレイ）
5. ✅ ログ確認：
   - `logs/mana_phase_debug.txt`: フラグが正しくプレイヤーごとに管理されているか
   - `logs/reset_turn_stats_debug.txt`: ターン開始時のリセット確認

### 動作確認後
6. Step 2の要否判断（TurnStats全体のプレイヤーごと管理）
7. Step 3の作業開始（PhaseManager移行）

---

## 参考情報

### 関連ファイル
- フラグ定義: `src/core/card_stats.hpp:65-74`
- フラグ設定: `src/engine/game_command/commands.cpp:545-556`
- フラグ参照: `src/engine/actions/strategies/phase_strategies.cpp:32`
- フェーズ管理: `src/engine/systems/flow/phase_manager.cpp`
- Python側: `dm_toolkit/commands.py`, `dm_toolkit/game_session.py`

### 削除済みPython側コード
- `_mana_charged_by_state` グローバル変数
- `_last_mana_count_by_state` グローバル変数
- マナチャージフィルタリングロジック（6箇所）
