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
- [ ] `action_converter.py` 互換シム削除（または空stub化）
- [ ] 旧データ読み込み時は migration 層でのみ吸収
- [ ] 全GUIテスト pass

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
- [ ] Filterの保存形式が明示的になる

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
- [ ] `python/tests/gui/test_signal_utils.py` に不足ケース追加:
  - signalはあるが `connect` 不可
  - widgetが `None`

### A-2 GREEN
- [ ] 対象フォームを3ファイルずつ置換:
  - `keyword_form.py`
  - `modifier_form.py`
  - `effect_form.py`

### A-3 REFACTOR
- [ ] `.connect(` の直書き残存を検索して優先度付け

実行コマンド:
```powershell
pytest python/tests/gui/test_signal_utils.py -q
pytest python/tests/gui/test_keyword_form.py -q
```

---

## フェーズB: 改善4（保存前整合チェック統合）

### B-1 RED
- [ ] `test_unified_action_validation.py` 新規作成
- [ ] 不正 `QUERY SELECT_OPTION` を保存しようとして失敗するテスト追加

### B-2 GREEN
- [ ] `unified_action_form.py` の保存処理に `validate_command_list` を統合
- [ ] 警告時のUIハイライト実装

### B-3 REFACTOR
- [ ] 警告文言の共通関数化

実行コマンド:
```powershell
pytest python/tests/gui/test_unified_action_validation.py -q
pytest tests/test_headless_editor.py -q
```

---

## フェーズC: 統合1+2（設定SSOT + ローダ統合）

### C-1 RED
- [ ] `test_editor_config_loader.py` 新規作成
- [ ] `command_ui.json` 読み込み経路が1つであることを検証

### C-2 GREEN
- [ ] `forms/command_config.py` のロード実装を `EditorConfigLoader` 呼び出しに委譲
- [ ] パス探索ロジック重複を削除

### C-3 REFACTOR
- [ ] `schema_config.py` の重複定義に TODO を付けて段階削減

---

## フェーズD: 改善5（ACTION完全削除）

### D-1 RED
- [ ] `grep` ベースで `"ACTION"` 参照一覧をテスト化（最終0件目標）

### D-2 GREEN
- [ ] `window.py`, `data_manager.py`, `normalize.py` から `ACTION` 分岐除去
- [ ] `action_converter.py` を削除または `CommandConverter` に改名して責務変更

### D-3 REFACTOR
- [ ] ドキュメント・コメントから Action表記を整理

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

### E-2 `FilterModel.flags` 非推奨化
- [ ] 読み込み時のみ `flags -> 明示フィールド` 変換
- [ ] 保存時は `flags` を出力しない

### E-3 変数リンクキー統一
- [ ] 保存キーを `input_value_key/output_value_key` に固定
- [ ] 旧キーは読取互換のみ

### E-4 `KeywordsModel` 導入
- [ ] `CardModel` の `keywords` を型モデル化
- [ ] 条件系キーを分離

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
- [ ] フェーズB 完了
- [ ] フェーズC 完了
- [ ] フェーズD 完了
- [ ] フェーズE 完了
- [ ] フェーズF 完了

最新メモ:
- 2026-03-14: `safe_connect` と `KeywordEditForm` の状態分離（`KeywordFormState`）導入済み。

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
