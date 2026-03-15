# カードエディタ改善・統合・データ構造再設計 実装計画

最終更新: 2026-03-15
対象: `dm_toolkit/gui/editor/` 一式
目的: 未完了タスクに集中し、小さな TDD サイクルで安全に改善を進める

---

## 0. 現状

- フェーズA〜Fの主要実装は完了済み。
- `safe_connect` 統一、保存前整合性チェック、設定SSOT化、主要モデル型付け、CIR最小統合までは到達済み。
- 直近で発生した統合プレビュー不整合は修正済み（`UnifiedActionForm.update_condition_preview` の導入とヘッドレス向けフォールバック実装）。
- 本ドキュメントは「未完了タスク管理」に用途を限定する。

注記:
- 完了済みの詳細ログ・検証ログは Git 履歴およびテスト履歴を参照。

---

## 1. 優先度付き残タスク

### 最優先

1. `window.py` の構造更新処理の分岐削減を完了する
- 目標: `on_structure_update()` と周辺処理を構造コマンド単位の完全ディスパッチへ統一する。
- 完了条件:
  - [x] 主要構造コマンド（追加/削除/移動/置換/CIR適用）の経路で if/elif チェーンを撤廃
  - [x] 共通後処理（選択同期、プレビュー更新、dirty state 更新）を1箇所に集約
  - [x] 回帰テストを追加/維持し、既存編集フローで退行がないことを確認

実装メモ (2026-03-15):
- `REPLACE_WITH_COMMAND` ハンドラから個別の選択同期・プレビュー更新を除去し、`on_structure_update()` の共通後処理へ統一。
- 回帰テスト `python/tests/unit/test_window_structure_update_replace_postprocess.py` を追加し、後処理が1回だけ実行されることを固定化。
- 関連テスト実行: 4 passed (`test_window_structure_update_replace_postprocess.py`, `test_window_replace_handler.py`, `test_window_dispatch.py`, `test_window_structure_handler_add_child_action.py`)

2. 分岐削減の定量追跡を継続する
- 目標: 分岐数削減を定期計測して、上位ファイルの改善優先度を維持する。
- 完了条件:
  - [x] `scripts/count_branches.py` の最新再計測を実施し、基準値との差分を記録
  - [x] 上位3ファイルの次回削減対象を明記（`text_generator.py`, `unified_action_form.py`, `logic_tree.py` 優先）
  - [x] 計測結果に基づく次スプリントの作業順を本書に反映

実装メモ (2026-03-15):
- 再計測コマンド: `.venv\\Scripts\\python scripts/count_branches.py`
- 計測結果: `Files with branches=331`, `if=4769`, `elif=606`, `total branches=5375`
- 基準値差分: `5375 -> 5375 (±0)`
  - 注記: ワークスペース内に過去計測ログが無かったため、今回値を基準値として固定。
- 次回削減対象（上位3）:
  - `dm_toolkit/gui/editor/text_generator.py` (907)
  - `dm_toolkit/gui/editor/forms/unified_action_form.py` (180)
  - `dm_toolkit/gui/editor/logic_tree.py` (69)
- 次スプリント作業順:
  1. `text_generator.py`: 既存ハンドラ辞書化の拡張（`_format_game_action_command` の残り分岐）
  2. `unified_action_form.py`: UI状態分岐の責務分割（表示制御と適用処理を分離）
  3. `logic_tree.py`: テンプレート適用・ノード追加系の条件分岐整理

### 高優先

3. 条件UIで `condition` / `trigger_filter` / `target_filter` の責務分離を完了する
- 目標: 発火条件・イベント対象条件・効果対象条件の編集文脈をUI上で明確化する。
- 完了条件:
  - [x] 3種の条件編集セクションを UI で分離表示
  - [x] 保存JSONのキー責務を定義どおりに固定（混在防止）
  - [x] テキスト生成とバリデーション双方でキー解釈が一致

実装メモ (2026-03-15):
- `EffectEditForm` の対象条件キーを `target_filter` に統一し、legacy `filter` は読込時のみ互換変換。
- モード別保存責務を固定:
  - `TRIGGERED` / `REPLACEMENT`: `trigger_filter` のみ保持（`target_filter`/`filter` を除去）
  - `STATIC`: `target_filter` のみ保持（`trigger_filter`/`filter` を除去）
- 共有バリデータを整合化:
  - `ModifierValidator`: `target_filter` 優先、`filter` 互換、両方異値はエラー
  - `TriggerEffectValidator`: `trigger_filter` を検証、`target_filter` 混入はエラー
- 回帰テスト追加: `python/tests/unit/test_effect_form_filter_role_separation.py`
- 関連テスト実行: `40 passed, 1 skipped`

4. 条件テンプレート機能を実装する
- 目標: 高頻度条件の入力をテンプレート化して入力ミスを減らす。
- 完了条件:
  - [x] テンプレート選択UIを追加（例: シールド枚数、相手ドロー枚数、ターン条件、文明数）
  - [x] 選択時に `ConditionEditorWidget` へ即時反映
  - [x] テンプレート適用の単体テストを追加

実装メモ (2026-03-15):
- `ConditionEditorWidget` にテンプレート選択UI（`template_combo`）を追加。
- テンプレート適用API `apply_template_by_key()` を追加し、選択時に即時反映（`set_data` + `dataChanged.emit`）を実装。
- 追加テンプレート:
  - 自分シールド3枚以下
  - 相手2枚目以降ドロー
  - 自分ターン中
  - 文明数2以上
- `MANA_CIVILIZATION_COUNT` を条件種別に追加し、UI表示設定を整備。
- 回帰対策: `NONE` 条件時にプレビューが「なし: 」だけ表示されないよう空プレビュー返却に修正。
- 単体テスト追加: `python/tests/unit/test_condition_templates.py`
- 関連テスト実行: `7 passed`

### 中優先

5. `CommandRegistry` を中心とした定義SSOT化を進める
- 目標: command定義分散（schema/config/validator/consts）を段階的に収束させる。
- 完了条件:
  - [x] registry 仕様（type/group/fields/validator/text hint）を確定
  - [x] 差分検出テストを CI 常設化
  - [x] 旧定義ファイルの参照先を整理し、二重管理を解消

実装メモ (2026-03-15):
- `schema_def.py` に `COMMAND_REGISTRY` を追加し、`register_schema()` 時に以下メタデータを同期登録する設計へ変更。
  - `type`, `group`, `fields`, `validator`, `text_hint`
- 追加API:
  - `get_registered_command_types()`
  - `get_command_registry_entry()`
  - `get_command_registry_snapshot()`
- 参照先整理:
  - `DynamicCommandForm` のコマンド種別ソースを `SCHEMA_REGISTRY` 直参照から `get_registered_command_types()` へ置換。
- 差分検出テスト追加（CI常設想定）:
  - `python/tests/unit/test_command_registry_ssot.py`
    - required spec fields の存在検証
    - schema registry と command_ui config の集合差分検証（既知差分を固定）
    - `SCHEMA_REGISTRY` との同期検証
- 関連テスト実行: `6 passed`

6. `text_generator.py` の `_format_game_action_command` 分岐削減を進める
- 目標: map 化済みハンドラを優先経路で実行し、重複した legacy `if/elif` を削減する。
- 完了条件:
  - [x] 後段で登録した `ACTION_HANDLER_MAP` ハンドラの再ディスパッチを追加
  - [x] 重複していた legacy 分岐（SEARCH/LOOK/PUT/SHUFFLE/BOOST/BREAK/SHIELD_BURN/REVEAL/COUNT など）を削除
  - [x] 分岐削減の回帰テストを追加

実装メモ (2026-03-15):
- `_format_game_action_command` に後段ハンドラ再ディスパッチを追加し、map 登録済みコマンドが legacy 分岐へ落ちないよう統一。
- 重複していた `elif atype == ...` ブロックを削除して分岐数を圧縮。
- 追加テスト: `python/tests/unit/test_text_generator_branch_reduction.py`
- 関連テスト実行: `12 passed`

7. `logic_tree.py` のテンプレート適用・ノード追加系分岐整理
- 目標: テンプレート適用後処理（選択更新・展開）と `payload.races` の文脈注入を共通化し、能力別メソッドの重複分岐を削減する。
- 完了条件:
  - [x] `add_rev_change` / `add_mekraid` / `add_friend_burst` / `add_mega_last_burst` を共通ヘルパー経由に統一
  - [x] `payload.races` 注入ロジックを共通化
  - [x] 分岐整理の回帰テストを追加

実装メモ (2026-03-15):
- `LogicTreeWidget` に `_build_races_context` と `_apply_logic_template` を追加し、テンプレート適用後の UI 更新を一本化。
- 上記4メソッドをヘルパー呼び出しへ置換し、重複していた `isValid` 判定・`apply_template_by_key` 呼び出し・`setCurrentIndex/expand` 処理を集約。
- `eff_item` 判定を `hasattr(get_raw_item)` ベースにして、QtEditorItem 互換オブジェクトでも後処理可能に改善。
- 追加テスト: `python/tests/unit/test_logic_tree_template_dispatch.py`
- 関連テスト実行: `3 passed`

8. `unified_action_form.py` のUI状態分岐整理（表示制御と適用処理の責務分離）
- 目標: `_load_ui_from_data` 内の CIR 表示制御を責務ごとに分離し、状態遷移の漏れを防ぐ。
- 完了条件:
  - [x] CIR 抽出・表示制御・差分描画をヘルパーへ分離
  - [x] CIR なし状態で Apply/Reject/Apply Selected と diff 表示を共通リセット
  - [x] 分岐整理の回帰テストを追加

実装メモ (2026-03-15):
- `UnifiedActionForm` に以下ヘルパーを追加。
  - `_extract_cir_entries(item)`
  - `_update_cir_ui_state(cir)`
  - `_update_cir_diff_view(cir)`
- `_load_ui_from_data` の CIR 統合分岐を上記ヘルパー呼び出しに置換し、表示制御と差分描画を分離。
- 再発防止として `_update_cir_ui_state` に CIR なし時の UI リセット（`apply_cir_btn` / `reject_cir_btn` / `apply_selected_btn` / `diff_tree_widget`）を集約。
- 追加テスト: `python/tests/unit/test_unified_action_form_cir_state_dispatch.py`
- 関連テスト実行: `4 passed`

---

## 2. 実行ルール

- 1回の実装は 1タスク・1症状・1〜3ファイル変更を原則とする
- 必ず `RED -> GREEN -> REFACTOR` で進める
- 実装後は関係する最小テストを優先実行し、必要に応じてフルテストを実行する
- エラー修正時は再発防止コメントを該当実装へ追加する

---

## 3. 次の着手候補

9. `text_generator.py` の次段分岐削減（`LOCK/RESTRICTION/STAT` 系のハンドラ辞書化）
- 目標: `LOCK_SPELL` / 制限系 / `STAT` 系を `ACTION_HANDLER_MAP` へ集約し、legacy `if/elif` 分岐を削減する。
- 完了条件:
  - [x] `LOCK_SPELL` と制限系（`SPELL_RESTRICTION` / `CANNOT_PUT_CREATURE` / `CANNOT_SUMMON_CREATURE` / `PLAYER_CANNOT_ATTACK`）を map ハンドラ化
  - [x] `STAT` / `GET_GAME_STAT` を map ハンドラ化
  - [x] 分岐削減の回帰テストを拡張

実装メモ (2026-03-15):
- `_format_game_action_command` に共通ヘルパー `_resolve_player_scope_text` / `_resolve_duration_text` を追加し、LOCK/制限系の対象プレイヤー・期間文言を統一。
- `ACTION_HANDLER_MAP` へ以下を登録:
  - `LOCK_SPELL`
  - `SPELL_RESTRICTION`, `CANNOT_PUT_CREATURE`, `CANNOT_SUMMON_CREATURE`, `PLAYER_CANNOT_ATTACK`
  - `STAT`, `GET_GAME_STAT`
- legacy 側の対応 `elif` ブロックを削除して分岐を圧縮。
- テスト更新: `python/tests/unit/test_text_generator_branch_reduction.py`
  - 除去対象 token に LOCK/RESTRICTION/STAT 系を追加
  - `LOCK_SPELL` / `SPELL_RESTRICTION` / `STAT` / `GET_GAME_STAT` の動作検証を追加
- 関連テスト実行: `3 passed`

10. `text_generator.py` の次段分岐削減（`FLOW/GAME_RESULT/DECLARE` 系のハンドラ辞書化）
- 目標: `FLOW` / `GAME_RESULT` / `DECLARE_NUMBER` / `DECIDE` / `DECLARE_REACTION` を `ACTION_HANDLER_MAP` へ集約し、legacy `if/elif` をさらに圧縮する。
- 完了条件:
  - [x] `FLOW` / `GAME_RESULT` を map ハンドラ化
  - [x] `DECLARE_NUMBER` / `DECIDE` / `DECLARE_REACTION` を map ハンドラ化
  - [x] 分岐削減の回帰テストを拡張

実装メモ (2026-03-15):
- `_format_game_action_command` に以下ハンドラを追加して `ACTION_HANDLER_MAP` へ登録。
  - `FLOW`
  - `GAME_RESULT`
  - `DECLARE_NUMBER`
  - `DECIDE`
  - `DECLARE_REACTION`
- 既存の legacy `elif` ブロックを削除し、既存文言を維持したまま分岐数のみ削減。
- テスト更新: `python/tests/unit/test_text_generator_branch_reduction.py`
  - 除去対象 token に FLOW/GAME_RESULT/DECLARE 系を追加
  - `FLOW` / `GAME_RESULT` / `DECLARE_NUMBER` / `DECIDE` / `DECLARE_REACTION` の動作検証を追加
- 関連テスト実行: `4 passed`

11. `text_generator.py` の次段分岐削減（`ATTACH/MOVE_TO_UNDER_CARD/RESET_INSTANCE/SELECT_TARGET` 系のハンドラ辞書化）
- 目標: 単純な対象操作コマンドを `ACTION_HANDLER_MAP` へ移し、legacy `if/elif` をさらに圧縮する。
- 完了条件:
  - [x] `ATTACH` / `MOVE_TO_UNDER_CARD` を map ハンドラ化
  - [x] `RESET_INSTANCE` / `SELECT_TARGET` を map ハンドラ化
  - [x] 分岐削減の回帰テストを拡張

実装メモ (2026-03-15):
- `_format_game_action_command` に以下ハンドラを追加して `ACTION_HANDLER_MAP` へ登録。
  - `ATTACH`
  - `MOVE_TO_UNDER_CARD`
  - `RESET_INSTANCE`
  - `SELECT_TARGET`
- 既存の legacy `elif` ブロックを削除し、対象文言・枚数文言は既存挙動を維持。
- テスト更新: `python/tests/unit/test_text_generator_branch_reduction.py`
  - 除去対象 token に ATTACH/MOVE_TO_UNDER_CARD/RESET_INSTANCE/SELECT_TARGET 系を追加
  - 4コマンドの動作検証を追加
- 関連テスト実行: `5 passed`

（未完了候補はこのタスク完了により現在なし）
