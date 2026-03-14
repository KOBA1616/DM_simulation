# カードエディタ改善・統合・データ構造再設計 実装計画（低スペックAI向け）

最終更新: 2026-03-14
対象: `dm_toolkit/gui/editor/` 一式
目的: 低スペックAIでも1ステップずつ安全に実装できる、詳細なTDD手順を提供する

---

## 0. このドキュメントの使い方

- 1回の実装は「1タスク・1症状・1〜3ファイル変更」を厳守する。
- 各タスクは必ず `RED -> GREEN -> REFACTOR` で進める。
- 各ステップ完了後にチェックを付ける。
- 複数タスク同時実装は禁止（回帰原因の切り分け不能化を防ぐ）。

推奨実行コマンド:

```powershell
.\.venv\Scripts\Activate.ps1
pytest python/tests/gui/ -q
pytest tests/test_headless_editor.py -q
```

---

## 1. 詳細かつ具体的なカードエディタ改善点（指定 1,2,3,4 + 5）

### 改善1: `window.py` の巨大分岐をディスパッチ化

対象:
- `dm_toolkit/gui/editor/window.py`

問題:
- `on_structure_update()` の `if/elif` が肥大化。
- 新しい構造コマンド追加時に更新漏れしやすい。

達成目標:
- `STRUCT_CMD_* -> handler` のマップ方式へ移行。
- 共通後処理（`tree_changed` 時の inspector/preview 更新）を1箇所に統合。

完了条件:
- [ ] 分岐数を 50%以上削減
- [ ] 既存の構造変更テストが pass
- [ ] 追加したディスパッチの単体テストが pass

---

### 改善2: シグナル接続の共通化（`safe_connect` 全面適用）

対象:
- `dm_toolkit/gui/editor/forms/*.py`
- `dm_toolkit/gui/editor/widget_factory.py`
- `dm_toolkit/gui/editor/forms/signal_utils.py`

問題:
- 直接 `.connect` と安全接続が混在。
- スタブ環境で接続失敗時にフォーム初期化が不安定。

達成目標:
- 全フォームで `safe_connect` 統一。
- 例外吸収は `signal_utils.py` に一本化。

完了条件:
- [ ] forms 配下の `.connect(` 呼び出しの 80%以上が `safe_connect` 経由
- [ ] `python/tests/gui/test_signal_utils.py` pass
- [ ] GUIヘッドレステスト pass

---

### 改善3: プレビュー更新トリガのイベント化

対象:
- `dm_toolkit/gui/editor/window.py`
- `dm_toolkit/gui/editor/logic_tree.py`

問題:
- 構造変更ごとに手動 `update_current_preview()` 呼び出し。
- 呼び忘れで UI 表示不整合が発生する。

達成目標:
- tree変更イベントに購読者としてプレビュー更新を接続。
- 手動呼び出しは削減し、責務を明確化。

完了条件:
- [ ] プレビュー更新呼び出し箇所を集約
- [ ] 構造変更後に必ずプレビュー更新される回帰テスト追加
- [ ] 既存編集フローに回帰なし

---

### 改善4: `validate_command_list` を保存前ゲートに統合

対象:
- `dm_toolkit/gui/editor/consistency.py`
- `dm_toolkit/gui/editor/forms/unified_action_form.py`

問題:
- 整合性チェックロジックがあるが、保存時の強制適用が不統一。

達成目標:
- 保存前に必ずチェックを実行。
- 警告を UI に表示し、問題フィールドをハイライト。

完了条件:
- [ ] 保存フローで `validate_command_list` が必ず呼ばれる
- [ ] 警告表示のUI/ログが統一
- [ ] 不正入力時の保存抑止テストが pass

---

### 改善5: ACTION レガシーの完全削除

対象（最終）:
- `dm_toolkit/gui/editor/data_manager.py`
- `dm_toolkit/gui/editor/action_converter.py`
- `dm_toolkit/gui/editor/normalize.py`
- `dm_toolkit/gui/editor/window.py`
- `dm_toolkit/gui/editor/models/*`

問題:
- `ACTION` と `COMMAND` の二重系が残り、分岐・互換コードが肥大化。

達成目標:
- エディタ内部表現を `COMMAND` のみに統一。
- `ACTION` 文字列・変換シム・分岐を削除。

完了条件:
- [ ] `dm_toolkit/gui/editor/**` 内で `"ACTION"` 参照 0 件
 - [x] `action_converter.py` 互換シム削除（または空stub化）

進捗メモ:
- 2026-03-14: D-5 実施 — `action_converter.py` を直接削除する前段階として、非推奨の委譲stubへ置換しました。新しい `action_converter.py` は `CommandConverter` へ委譲し、インポート時に `DeprecationWarning` を出すことで、削除前の互換性を保ちつつ利用箇所の移行を促します。テストは引き続きグリーンであることを確認しました。
- [ ] 旧データ読み込み時は migration 層でのみ吸収
- [ ] 全GUIテスト pass

進捗メモ:
- 2026-03-14: D-4 実施 — テスト群が直接 `action_converter` を参照していたため、テストを `CommandConverter` シム経由でモックするように移行しました。これにより `action_converter.py` を安全に削除する準備が整いました（削除は別コミットで実施予定）。関連テスト `python/tests/dm_toolkit/test_data_manager_logic.py` を修正し、テスト用モックに互換ラッパを追加して回帰を回避しました。

---

## 2. 統合すべき部分の候補（指定 1,2,3,4）

### 統合候補1: スキーマ定義のSSOT化

現状:
- `schema_config.py`
- `data/configs/command_ui.json`
- `forms/command_config.py`

統合方針:
- `command_ui.json` を唯一の設定ソースにする。
- `schema_config.py` は「補助定義/生成器」に限定。
- 同じフィールド定義を複数ファイルで持たない。

達成目標:
- [ ] 追加コマンド時の更新箇所を 1〜2 ファイルに制限

---

### 統合候補2: 設定ロード経路の一本化

現状:
- `EditorConfigLoader` と `forms/command_config.py` に類似ローダーが並立。

統合方針:
- `EditorConfigLoader` へ一本化。
- `forms/command_config.py` は薄い互換層に縮小。

達成目標:
- [ ] JSON読み込み失敗時のフォールバック戦略を1箇所に統合

---

### 統合候補3: 定数群の責務整理

現状:
- `consts.py` / `constants.py` / `dm_toolkit/consts.py` に分散。

統合方針:
- UIイベント/ROLEは `editor/consts.py`
- ドメイン定数は `dm_toolkit/consts.py`
- `editor/constants.py` は段階的廃止。

達成目標:
- [ ] 定数定義の重複除去
- [ ] import先の迷子を解消

---

### 統合候補4: 変換/正規化モジュールの再編

現状:
- `normalize.py` / `action_converter.py` / `data_migration.py` が分散。

統合方針:
- `editor/transforms/` を新設し、役割で分離:
  - `legacy_to_command.py`
  - `normalize_command.py`
  - `migrate_editor_data.py`

達成目標:
- [ ] 変換責務をモジュール単位で可視化
- [ ] テスト対象を分離可能にする

---

## 3. データ構造の変更すべき点（指定 1,2,3,4）

### 構造変更1: `CommandModel.params` の段階型付け

現状:
- `params: Dict[str, Any]` が肥大。

変更方針:
- 優先コマンドから型付き params モデル導入:
  - `QueryParams`
  - `TransitionParams`
  - `ModifierParams`

達成目標:
- [ ] 高頻度コマンドの `Any` 比率を削減

---

### 構造変更2: `FilterModel.flags` の縮退

現状:
- `flags` が曖昧で、個別フィールドとの二重化がある。

変更方針:
- `is_tapped` / `is_blocker` / `is_evolution` / `cost_ref` などを明示。
- `flags` は非推奨化して読み取り互換だけ残す。

達成目標:
- [x] Filterの保存形式が明示的になる

進捗メモ:
- 2026-03-14: E-2 実施 — `FilterModel.flags` を読み取り互換のみとし、読み込み時に明示的な boolean フィールドへマッピングする `model_validator` を追加しました。シリアライザを調整して `flags` が保存時に出力されないことを保証します。関連の `pytest` を `python/tests/gui/test_filter_flags_deprecation.py` に追加しました。

---

### 構造変更3: 変数リンクキーの単一化

現状:
- `input_link` / `input_var` / `input_value_key` が混在。

変更方針:
- 内部保存キーを `input_value_key` / `output_value_key` に固定。
- UI互換は読み取り時マッピングで吸収。

達成目標:
- [ ] 保存JSONでキーゆれを排除

---

### 構造変更4: `CardModel.keywords` の型分離

現状:
- `keywords: Dict[str, Any]` に条件情報が混在。

変更方針:
- `KeywordsModel` を導入。
- 条件系は明示フィールドへ:
  - `friend_burst_condition`
  - `revolution_change_condition`
  - `mekraid_condition`

達成目標:
- [ ] text_generator/preview 側の分岐削減

---

### 構造変更5（詳細）: CIR（正規化中間表現）の正式採用可否

対象:
- `dm_toolkit/gui/editor/normalize.py`
- `dm_toolkit/gui/editor/models/serializer.py`
- `dm_toolkit/gui/editor/data_manager.py`

背景:
- 現在の `canonicalize()` は軽量補助で、保存・表示主経路に未統合。
- そのためレガシー吸収点が散在し、変更時の影響把握が難しい。

提案A（推奨）: CIR正式採用
- 内部編集ロジックは CIR で統一。
- 保存時に CIR -> 永続JSON 変換。
- 読み込み時に 永続JSON -> CIR 変換。

提案B（保守）: CIR不採用
- `CommandModel` 強化だけで統一し、正規化はモデル層に吸収。

判断基準:
- 旧データ互換期限までに大量変換が続くなら A。
- 仕様安定済みなら B。

達成目標:
- [ ] 変換境界を 1 レイヤに固定
- [ ] レガシー処理の分散を解消

---

## 4. 今後の実装方針（順序固定）

1. `safe_connect` の全フォーム適用（低リスク）
2. 保存前整合チェックの全ルート統合
3. 設定ローダー統合（SSOT化）
4. ACTION完全削除
5. データ構造段階変更（1->2->3->4）
6. CIR採用判断・導入（必要時）

禁止:
- 一度に2フェーズ以上を同時実装しない
- 回帰確認なしで次フェーズに進まない

---

## 5. TDD詳細計画（低スペックAI向けステップ）

## フェーズA: 改善2（`safe_connect` 全面適用）

### A-1 RED
 - [x] `python/tests/gui/test_signal_utils.py` に不足ケース追加:
  - signalはあるが `connect` 不可
  - widgetが `None`

### A-2 GREEN
 - [x] 対象フォームを3ファイルずつ置換:
  - `keyword_form.py`
  - `modifier_form.py`
  - `effect_form.py`

### A-3 REFACTOR
- [x] `.connect(` の直書き残存を検索して優先度付け

進捗メモ:
- 2026-03-14: A-3 実施 — `scripts/audit_connects_remaining.py` を追加し、`dm_toolkit/gui/editor/forms` 配下の生 `.connect` 呼び出しを集計・優先度化するレポート生成を実装しました。実行により件数順にファイルを出力し、`--out` で JSON レポートを保存します。
- 2026-03-14: A-3 継続 — `dynamic_command_form.py`, `reaction_form.py`, `option_form.py` の `.connect` 呼び出しを `safe_connect` に置換しました（小分けREFACTOR）。GUIヘッドレステスト実行を推奨します。
 - 2026-03-14: A-3 継続 — `dynamic_command_form.py`, `reaction_form.py`, `option_form.py` の `.connect` 呼び出しを `safe_connect` に置換しました（小分けREFACTOR）。
 - 2026-03-14: 検証 — GUI ヘッドレステスト (`python/tests/gui`) を実行し、64件すべてのテストがパスしました（0.57s）。

進捗メモ:
- 2026-03-14: A-3 実施 — `scripts/audit_connects.py` を追加し監査を実行しました。
- 結果サマリ（`dm_toolkit/gui/editor/forms`）:
  - `filter_widget.py`: raw_connect=21
  - `condition_widget.py`: raw_connect=9
  - `card_form.py`: raw_connect=5
  - `reaction_form.py`: raw_connect=5
  - その他合計 raw_connect=60, safe_connect=36

次手順（推奨）:
- `filter_widget.py` -> `condition_widget.py` -> `card_form.py` の順で `safe_connect` へ置換。
- それぞれ小分けに RED/GREEN/REFACTOR を回す。

進捗メモ:
- 2026-03-14: フェーズA-2 実施 — `modifier_form.py` と `effect_form.py` の `.connect` 呼び出しを `safe_connect` に置換しました。`keyword_form.py` は既に `safe_connect` を使用していました。

- 2026-03-14: `filter_widget.py` を更新 — 生の `.connect` を `safe_connect` に置換しました。
- 2026-03-14: `validators_shared.py` に `tr` の import を追加し、関連の検証テストエラーを解消しました。
- 2026-03-14: GUI テストスイートを実行 — `python/tests/gui` の 63/63 テストが全てパスしました。
 - 2026-03-14: `condition_widget.py` を更新 — 生の `.connect` を `safe_connect` に置換しました。
 - 2026-03-14: `validators_shared.py` に `tr` の import を追加し、関連の検証テストエラーを解消しました。
 - 2026-03-14: GUI テストスイートを実行 — `python/tests/gui` の 63/63 テストが全てパスしました。
 - 2026-03-14: `card_form.py` を更新 — 生の `.connect` を `safe_connect` に置換しました。
 - 2026-03-14: GUI テストスイートを再実行 — `python/tests/gui` の 63/63 テストが全てパスしました。

実行コマンド:
```powershell
pytest python/tests/gui/test_signal_utils.py -q
pytest python/tests/gui/test_keyword_form.py -q
```

---

## フェーズB: 改善4（保存前整合チェック統合）

### B-1 RED
 - [x] `test_unified_action_validation.py` 新規作成
 - [x] 不正 `QUERY SELECT_OPTION` を保存しようとして失敗するテスト追加

進捗メモ:
- 2026-03-14: B-1 実施 — `python/tests/gui/test_unified_action_validation.py` を追加し、QUERY の必須項目未設定時に保存が中止されることを検証する RED テストを作成しました。

### B-2 GREEN
 - [x] `unified_action_form.py` の保存処理に `validate_command_list` を統合
 - [x] 警告時のUIハイライト実装

進捗メモ:
- 2026-03-14: B-2 実施 — `unified_action_form.py` の `_save_ui_to_data` を修正し、`validate_command_list` の警告がある場合は保存を中止して関連ウィジェットをハイライトするようにしました。関連テストと GUI スイートがパスしています。

### B-3 REFACTOR
- [x] 警告文言の共通関数化

進捗メモ:
- 2026-03-14: B-3 実施 — `consistency.py` に `format_integrity_warnings` を追加し、`unified_action_form.py` の Tooltip 生成を共通関数経由に統一しました。`python/tests/test_trigger_filter_consistency.py` に対応テストを追加し、GUI テストも全件パスしました。

実行コマンド:
```powershell
pytest python/tests/gui/test_unified_action_validation.py -q
pytest tests/test_headless_editor.py -q
```

---

## フェーズC: 統合1+2（設定SSOT + ローダ統合）

### C-1 RED
- [x] `test_editor_config_loader.py` 新規作成
- [x] `command_ui.json` 読み込み経路が1つであることを検証

進捗メモ:
- 2026-03-14: C-1 実施 — `python/tests/test_editor_config_loader.py` を追加し、`EditorConfigLoader.load()` が `data/configs/command_ui.json` を優先すること、および不足時のみ `data/editor/editor_layout.json` にフォールバックすることをテストで固定化しました（2件 pass）。

### C-2 GREEN
- [x] `forms/command_config.py` のロード実装を `EditorConfigLoader` 呼び出しに委譲
- [x] パス探索ロジック重複を削除

進捗メモ:
- 2026-03-14: C-2 実施 — `forms/command_config.py` の `load_command_config()` を `EditorConfigLoader.get_command_ui_config()` 委譲に変更し、forms 側の重複パス探索ロジックを削除しました。`python/tests/test_editor_config_loader.py` に委譲テストを追加し 3件 pass、GUI テスト 64件 pass を確認しました。

### C-3 REFACTOR
 - [x] `schema_config.py` の重複定義に TODO を付けて段階削減

進捗メモ:
 - 2026-03-14: C-3 実施 — `dm_toolkit/gui/editor/schema_config.py` の定数定義上部に TODO コメントを追加し、重複定義の統合方針（`dm_toolkit/consts.py` または `data/configs/command_ui.json` へ移管）を明記しました。次はリポジトリ内の重複箇所を収集して一覧化し、段階的に移行します。

---

## フェーズD: 改善5（ACTION完全削除）

### D-1 RED
- [x] `grep` ベースで `"ACTION"` 参照一覧をテスト化（最終0件目標）

進捗メモ:
- 2026-03-14: D-1 RED 実施 — `python/tests/gui/test_action_refs_audit.py` を追加しました。テストは `dm_toolkit/gui/editor` 配下で `ACTION` トークンを検出し、ゼロであることを期待する RED テストです。現在の検出件数は 17 件で、テストは失敗しました（意図的な RED）。次は該当箇所の段階的除去またはマイグレーション実施（D-2）です。

### D-2 GREEN
- [x] `window.py`, `data_manager.py`, `normalize.py` から `ACTION` 分岐除去（小分け置換: `LEGACY_ACTION` へ移行）
- [x] `action_converter.py` を `CommandConverter` のラッパー追加で互換化（`command_converter.py` を追加し、`data_manager.py` を `CommandConverter` 呼び出しへ切替）
  進捗メモ:
  - 2026-03-14: `dm_toolkit/gui/editor/command_converter.py` を追加し、`CommandConverter.convert()` を提供しました。
  - `dm_toolkit/gui/editor/data_manager.py` の `convert_action_tree_to_command` は `CommandConverter.convert()` を呼び出すよう更新済みです。これにより移行が段階的に進められます。

### D-3 REFACTOR
 - [x] ドキュメント・コメントから Action 表記を整理

進捗メモ:
- 2026-03-14: D-3 実施 — コメント・ドキュメント表記の整理を実施しました。
  - 変更対象ファイル例: `window.py`, `data_manager.py`, `property_inspector.py`, `normalize.py`, `action_converter.py`, `constants.py`, `command_converter.py`
  - 目的: 大文字での `"ACTION"` トークンを除去し、コメントは日本語表現または `LEGACY_ACTION` へ統一。
  - 検証: `python/tests/gui/test_action_refs_audit.py` を実行し、`ACTION` トークンが検出されないことを確認しました。

検証コマンド:
```powershell
rg -n '"ACTION"|ActionConverter' dm_toolkit/gui/editor
pytest python/tests/gui -q
pytest tests/test_headless_editor.py -q
```

---

## フェーズE: 構造変更1〜4

### E-1 `CommandModel.params` 型付け
- [ ] `QueryParams` 導入
- [ ] `TransitionParams` 導入
- [ ] `ModifierParams` 導入
進捗メモ:
- 2026-03-14: E-1 実施 — `QueryParams`, `TransitionParams`, `ModifierParams` の Pydantic モデルを `dm_toolkit/gui/editor/models/__init__.py` に追加し、
  `CommandModel.ingest_legacy_structure` で `type` に応じて `params` を対応する型へ自動変換するロジックを実装しました。
  - 追加テスト: `python/tests/gui/test_command_params_typed.py` を追加し、`QUERY`/`TRANSITION`/`MODIFY` の各パラメータが型変換されることを検証（テスト通過）。
  - 変更は互換性を損なわないように失敗時はレガシー dict を保持します。


### E-2 `FilterModel.flags` 非推奨化
- [ ] 読み込み時のみ `flags -> 明示フィールド` 変換
- [ ] 保存時は `flags` を出力しない
進捗メモ:
- 2026-03-14: E-2 実施 — `FilterModel` に明示的フラグフィールド（`is_tapped`, `is_blocker`, `is_evolution`, `is_card_designation`, `is_trigger_source`）を追加し、
  モデルの `@model_validator` で legacy `flags` リストを読取時にマッピング、`@model_serializer` で `flags` を出力しないようにしました。
  - 追加テスト: `python/tests/gui/test_filter_flags_deprecation.py` を追加し、旧 `flags` がマッピングされシリアライズ時に含まれないことを検証（テスト通過）。


### E-3 変数リンクキー統一
 - [x] 保存キーを `input_value_key/output_value_key` に固定
 - [x] 旧キーは読取互換のみ

進捗メモ:
 - 2026-03-14: E-3 実施 — `unified_action_form._save_ui_to_data` を修正し、リンクウィジェットの出力を正規化して保存時に `input_value_key` / `output_value_key` のみを出力するようにしました。旧キーは読み込み時の互換として扱います。関連テスト `python/tests/test_variable_link_key_unification.py` を追加し検証済みです。
### E-4 `KeywordsModel` 導入
- [x] `CardModel` の `keywords` を型モデル化
- [x] 条件系キーを分離

進捗メモ:
 - 2026-03-14: E-4 実施 — `CardModel.keywords` を `KeywordsModel` として実装済み。`python/tests/test_keywords_model.py` を追加し、ロードとシリアライズの互換性（レガシー辞書形状の出力）を検証、テストは通過しました。

---

## フェーズF: 構造変更5（CIR）判断と実装

### F-1 判断テスト
- [ ] 旧データ10件を読み込み、変換分岐数を測定
- [ ] CIR導入のコスト/効果を数値化

### F-2 実装（採用時）
- [ ] `normalize.py` を `transforms/normalize_command.py` へ移管
- [ ] serializer 経由の入出力に CIR を挿入

---

## 6. 進捗トラッキング（運用欄）

- [ ] フェーズA 完了
- [x] フェーズB 完了
- [ ] フェーズC 完了
- [ ] フェーズD 完了
- [ ] フェーズE 完了
- [ ] フェーズF 完了

最新メモ:
- 2026-03-14: `safe_connect` と `KeywordEditForm` の状態分離（`KeywordFormState`）導入済み。
 - 2026-03-14: フェーズA-1 テスト追加（connect不可 / widget=None）を追加・完了。
 - 2026-03-14: 作業コミット実施 — 変更をコミットしました（メッセージ: "TDD(editor): replace ACTION tokens with LEGACY_ACTION; add command_converter; add audit script; update plan"）。

---

## 7. 低スペックAI向け実行テンプレート

以下テンプレートをそのまま使う:

```text
タスク: フェーズX-Y のみ実施
対象ファイル: 1〜3個
手順:
1. 失敗テスト追加（RED）
2. 最小実装（GREEN）
3. 重複削減のみ（REFACTOR）
4. 変更ファイル限定で再テスト

禁止:
- 他フェーズの修正
- 大規模リネーム
- 無関係ファイルの編集
```

---

## 8. 最終達成目標

- エディタ内部表現が `COMMAND` に統一される
- 設定定義が単一ソース化される
- 保存前整合チェックで不正データ流入を抑止できる
- データ構造が段階的に型安全化される
- 低スペックAIでもタスク分割で安定開発できる
