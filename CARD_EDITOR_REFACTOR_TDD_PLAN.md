# カードエディタ改善・統合・データ構造再設計 実装計画

最終更新: 2026-03-16
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

## 1. 根本改善バックログ（未完了のみ）

### 最優先: コスト軽減基盤の一元化

1. コスト判定・支払い実行を単一路線化する
- 背景:
  - `ManaSystem` と `CostPaymentSystem` に責務が分散しており、判定系と実行系の乖離が再発しやすい。
- 目標:
  - `can_pay` と `execute` が同一の計算結果を共有する `PaymentPlan` ベースへ移行する。
- 完了条件:
  - [x] Python プロトタイプ: `evaluate_cost(...) -> PaymentPlan` を導入（エンジン C++ 実装は次フェーズ）
  - [x] C++ 設計/プロトタイプ: `docs/design/payment_plan_design.md` と `src/engine/systems/mechanics/payment_plan.hpp/.cpp` を追加（エンジン統合は次フェーズ）
    - [x] エンジン実装追記: `evaluate_cost` を `id` 優先（`name` フォールバック）で照合するように更新（`src/engine/systems/mechanics/payment_plan.cpp`）
  - [ ] 判定経路（合法手生成）と実行経路（`PLAY_FROM_ZONE`）が同一 `PaymentPlan` を利用
  - [ ] 既存の重複計算（`ManaSystem`/`CostPaymentSystem` の二重判定）を段階的に解消

2. 常在コスト軽減のデータソースを統合する
- 背景:
  - `static_abilities(COST_MODIFIER)` は有効だが、`cost_reductions(PASSIVE)` は実運用で反映されない。
- 目標:
  - 常在軽減の仕様を1本化し、実装で取りこぼしが発生しない状態にする。
- 完了条件:
  - [ ] `PASSIVE` 軽減の取り扱いを仕様として明文化（優先順位/合成ルール）
  - [x] 実装側で `PASSIVE` も支払い計算へ反映（Python 側プロトタイプを追加、単体テストあり）
  - [x] 既存カードデータの互換マイグレーション方針を策定およびスクリプトを追加（`scripts/migrate_cost_reduction_ids.py`）

3. 支払いコマンドのデータ構造を構造化する
- 背景:
  - 現状の `str_param`/`str_val`/`target_instance` 依存は可読性と保守性が低い。
- 目標:
  - 支払い意図を明示するフィールドへ移行し、UI/AI/実行系の解釈ズレを防止する。
- 完了条件:
 - 完了条件 (進捗):
  - [x] `CommandDef` に支払い専用フィールド（`payment_mode`, `reduction_id`, `payment_units`）を追加（エディタ/モデル側）
  - [ ] 生成系・実行系・バインディングの対応を完了
  - 実装ノート: エディタ/生成側での対応を段階的に進めています。
    - [x] `dm_toolkit.command_builders.build_play_card_command` と `_build_native_command` に `payment_mode`/`reduction_id`/`payment_units` のマッピングを追加（ネイティブ/辞書双方を通じて伝搬可能）
    - [x] 単体テスト `tests/test_command_builders_payment_fields.py` を追加し、非ネイティブ辞書生成でフィールドが含まれることを確認
    - [x] 単体テスト `tests/test_build_native_command_payment_fields.py` を追加し、ネイティブビルド経路でも `payment_*` フィールドが `CommandDef` 相当オブジェクトへマッピングされることを確認
    - [ ] エンジン側（C++ バインディング / `CommandDef` 実体）の受け取りと実行系での利用は未完了（次フェーズ）
  - [ ] 既存形式との後方互換（ランタイム実行系・AI 生成経路での取り込み）を実装

実装ノート:
- `dm_toolkit/gui/editor/models/CommandModel` に `payment_mode`, `reduction_id`, `payment_units` を追加し、`ingest_legacy_structure` とシリアライズで扱うようにしました。
- 単体テスト `tests/test_commandmodel_payment_fields.py` を追加し、フィールドがモデルの構築/ダンプで保持されることを確認しました。


### 高優先: カードエディタとデータ品質の強化

4. カードエディタに `cost_reductions` 専用編集導線を追加する
- 背景:
  - 現状はテキスト生成中心で、編集・検証の専用UXが不足している。
- 目標:
  - PASSIVE/ACTIVE_PAYMENT の編集をフォームで完結できる状態にする。
- 完了条件:
  - [x] `cost_reductions` 編集UI（追加/削除/並び替えの基本操作）を実装（簡易 JSON 編集付き）
  - [x] `unit_cost/filter/min_mana_cost/max_units` の入力補助を編集ウィジェット API として追加（ヘッドレス/テスト用 API 実装）
  - 実装ノート: `CostReductionEditor` に `set_selected_index` / `update_selected_fields` を追加しました。これにより UI 側から個別エントリの `amount`/`min_mana_cost`/`unit_cost`/`max_units` を設定でき、テストではヘッドレス実装を用いて動作確認済みです。GUI 上の専用入力コントロール（スピンボックス等）へのマッピングは次フェーズで追加します。
  - [x] GUI 側の入力コントロール（`QSpinBox`）を CostReductionEditor に追加し、スピンボックス操作が選択エントリに反映されるようにバインドしました（環境によりスキップされる GUI テストあり）。
  - 実装ノート: GUI のコントロール追加は `dm_toolkit/gui/editor/forms/parts/cost_reduction_editor.py` の Qt 実装側に追加しました。ヘッドレスフォールバックはそのまま維持しています。
  - [x] `unit_cost/filter/min_mana_cost/max_units` の入力補助を実装
  - [x] プレビューに `units -> effective_cost` 試算表示を追加

UI 拡張:
 - `CostReductionEditor` の Qt 実装にプレビュー表示用の `QLabel` を追加しました。選択変更・JSON編集・入力補助値変更時にラベルを更新することで、即時フィードバックを提供します。
 - GUI 単体テスト `tests/test_cost_reduction_preview_gui.py` を追加しました。PyQt6 非搭載環境では自動でスキップされます。

入力補助:
 - `suggest_input_assist(context=None)` をヘッドレス/Qt 双方に追加しました。返す値は `unit_cost`/`max_units`/`min_mana_cost` の推奨値で、既存値を尊重し未指定時は簡潔なヒューリスティックで導出します。
 - ヘッドレス単体テスト `tests/test_cost_reduction_input_assist.py` を追加し、基本導出ロジックを検証済みです。

実装ノート:
 - `CostReductionEditor`（ヘッドレス/Qt 双方）に `compute_effective_cost(units=None)` と `get_preview_text(units=None)` を追加し、エディタ上での簡易試算を行えるようにしました。
 - ヘッドレス単体テスト `tests/test_cost_reduction_preview.py` を追加し、基本ケース（unit_cost・max_units・min_mana_cost の組合せ）を検証済みです。

実装ノート:
 - `dm_toolkit/gui/editor/forms/parts/cost_reduction_editor.py` を追加し、左に項目一覧（ドラッグ/並び替え対応）、右に JSON エディタを備えた簡易編集ウィジェットを実装しました（ヘッドレス環境向けの純 Python フォールバックあり）。
 - `CardEditForm` に統合し、フォームの `cost_reductions` フィールドとして `get_value`/`set_value` を通じて保存/読み込みに対応します。
 - 単体テスト `tests/test_cost_reduction_editor_widget.py` を追加し、`set_value`/`get_value` のラウンドトリップと `id` 自動生成を検証しています（ヘッドレスでの実行を想定し、フォールバック実装を使用）。

5. `cost_reductions` スキーマを強化する
- 背景:
  - `name` 依存は曖昧性が残り、複数軽減定義の識別に弱い。
- 目標:
  - 実行識別に使う一意IDを導入し、表示名と分離する。
- 完了条件:
  - [x] `cost_reductions[].id`（必須・一意）を導入
  - [ ] `name` は表示専用に整理
  - [x] ローダー/シリアライザ/エディタの保存検証で重複IDをエラー化

6. 保存時バリデーションを追加する (エディタ側実装: 完了)
- 背景:
  - 不完全な軽減定義が保存されると実行時エラーや無効化が発生する。
- 目標:
  - 実行不能なデータを保存時点で検出し、早期に修正可能にする。
- 完了条件 (進捗):
  - [x] `ACTIVE_PAYMENT` 必須項目欠落の検出 (エディタ保存時に検出し、UI にフィードバックして保存を中止します)
  - [x] `reduction_amount <= 0` や不正 `min_mana_cost` の検出 (数値チェックは `validators_shared` に実装され、保存時に適用されます)
  - [x] `PASSIVE` と `static_abilities(COST_MODIFIER)` の競合警告 (エディタ側で検出・警告を実装済み)

実装ノート:
- `dm_toolkit/gui/editor/validators_shared.py::detect_passive_static_conflicts` を追加し、PASSIVE と COST_MODIFIER の同時定義を検出して警告を返すユーティリティを実装しました。
- `BaseEditForm.save_data` の保存前検証でこの警告を参照して UI フィードバックを出すことも可能です（既に cost_reductions 検証との連携は行っています）。
 - 実装ノート:
 - `dm_toolkit/gui/editor/validators_shared.py::detect_passive_static_conflicts` を追加し、PASSIVE と COST_MODIFIER の同時定義を検出して警告を返すユーティリティを実装しました。
 - `BaseEditForm.save_data` に統合し、競合が検出された場合は**保存を中止せずに**フォームのバインドウィジェットへツールチップで警告を表示するようにしました（非ブロッキング警告として扱います）。

実装ノート:
- `dm_toolkit/gui/editor/validators_shared.py` に検証ロジックを実装。
- `dm_toolkit/gui/editor/forms/base_form.py::BaseEditForm.save_data` に保存前検証を統合し、検証エラー時はフォームのバインドウィジェットへスタイル/ツールチップでフィードバックを与え、保存処理を中止するようにしました。
- テスト: `tests/test_editor_save_validation.py` を追加し、保存中止と UI フィードバック挙動を検証（ヘッドレスモードでの動作確認用にテストはダミー Qt を強制）。

### 中優先: テスト戦略の再構築

7. コスト軽減仕様の回帰テストマトリクスを整備する
- 背景:
  - 現在のテストは `cost_reductions` のロード耐性確認が中心で、実プレイ網羅が不足している。
- 目標:
  - 常在/能動軽減の主要ケースを自動テストで固定化する。
- 完了条件:
  - [x] `PASSIVE` 軽減の適用/非適用テスト
  - [x] `ACTIVE_PAYMENT` の units 別コスト計算テスト
  - [x] 文明不足・`min_mana_cost` クランプ・複数軽減共存テスト（文明不足テストを実装・追加）
  - [ ] C++/Python 統合経路での回帰テストを追加

---

## 2. 実行ルール

- 1回の実装は 1タスク・1症状・1〜3ファイル変更を原則とする
- 必ず `RED -> GREEN -> REFACTOR` で進める
- 実装後は関係する最小テストを優先実行し、必要に応じてフルテストを実行する
- エラー修正時は再発防止コメントを該当実装へ追加する

---

## 3. 次の着手候補（短期スプリント）

1. `PaymentPlan` の最小導入
- 対象: `src/engine/systems/mechanics/mana_system.cpp`, `src/engine/systems/mechanics/cost_payment_system.cpp`, `src/engine/systems/director/game_logic_system.cpp`
- 内容: `ACTIVE_PAYMENT` を含む支払い判定結果を構造体で返し、実行経路へそのまま受け渡す。

2. `cost_reductions.id` の導入と互換読込 (エディタ側 + モデル + マイグレーション対応: 完了)
- 実行済み（エディタ側ユーティリティ）: `dm_toolkit/gui/editor/validators_shared.py` に `generate_missing_ids` を追加し、保存前に欠損 `id` を自動付与するユーティリティを導入しました。
- 保存時統合: `dm_toolkit/gui/editor/models/serializer.py::save_full_data` に統合し、実際に保存される JSON に `id` を付与するようにしました。
- モデル側対応: `dm_toolkit/gui/editor/models/__init__.py` に `CostReductionModel` を導入し、`CardModel` の `cost_reductions` を `List[CostReductionModel]` として厳格化しました。さらに `CardModel` の `model_validator` で読込時に `generate_missing_ids` を呼び、既存データの互換性を維持します。
- マイグレーション対応: `dm_toolkit/gui/editor/data_migration.py` に `migrate_cost_reductions` を追加し、ロード/バッチ移行時に欠損 `id` を付与できるようにしました。これによりエディタ外部での一括移行作業が容易になります。
- 対象（今後）: `src/core/card_json_types.hpp`, `src/engine/infrastructure/data/json_loader.cpp` にも同等の後方互換処理を導入する計画です。
- 実装追記: C++ 側 `CostReductionDef` に `id` フィールドを追加し、`json_loader.cpp` 側で欠落 `id` を `cr_<card_id>_<index>` 形式で自動付与するプロトタイプを導入しました（エンジン統合の本実装は次フェーズ）。
- テスト: 以下のテストを追加しました。
  - `tests/test_cost_reduction_id_migration.py` : 欠損 id の割当と既存 id の保持を確認（2 件パス）。
  - `tests/test_save_full_data_auto_ids.py` : `ModelSerializer.save_full_data` が保存前に id を付与して JSON に書き出すことを検証（1 件パス）。
  - `tests/test_cardmodel_cost_reduction_id.py` : `CardModel` の構築時に欠損 id が自動で付与されることを検証（1 件追加）。

注意: C++ ローダー側の互換処理（既存 JSON を読み込む際の自動付与）は別途実装予定です。

追記: `src/engine/infrastructure/data/json_loader.cpp` に後方互換処理を追加しました。`cost_reductions` の `name` が空の場合、ロード時に `cr_<card_id>_<index>` 形式の安定した識別子を自動付与します（これによりエンジン側での互換性が向上します）。

3. エディタ側バリデーションの先行実装 (完了)
- 実行済み: `dm_toolkit/gui/editor/validators_shared.py` を追加し、基本的なスキーマ検証を実装しました。
- 対象: `dm_toolkit/gui/editor/validators_shared.py`, `dm_toolkit/gui/editor/forms/*`
- 内容: `ACTIVE_PAYMENT` の必須項目チェック、数値範囲チェック、重複 `id` 検出を実装。保存前呼び出しは引き続きエディタ側統合が必要ですが、単体検証ユーティリティは導入済みです。
- テスト: `tests/test_cost_reduction_validator.py` を追加し、基本ケースの回帰テストを導入しました（3件すべてパス）。

### 実行記録: C++ ビルド/テスト試行

- 日時: 2026-03-16
- 内容: エージェントがリポジトリ内で C++ の構成・ビルド・`ctest` 実行を試行しました（`-DENABLE_CPP_TESTS=ON`）。
- 結果: 失敗
  - 要因: ビルド環境に C++ コンパイラが見つからず CMake が構成できませんでした。
  - エラーメッセージ抜粋: `CMake Error: No CMAKE_CXX_COMPILER could be found.`
- 推奨対応:
  - Windows 環境では Visual Studio の "Desktop development with C++" ワークロードまたは Build Tools をインストールしてください。
  - 代替で MSYS2/MinGW の g++ を導入して `-G Ninja` と組み合わせることも可能です。
  - ツールチェイン導入後に以下を実行して再検証してください:

```powershell
cmake -S . -B build -G Ninja -DENABLE_CPP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build -V
```

追記: CI 側でも同様のツールチェインが必要になります。環境整備後、改めてビルド/テストを実行します。

### 実行記録: data/ 配下の JSON 一括移行 (スクリプト実行)

- 日時: 2026-03-16
- 実施内容: `scripts/migrate_cost_reduction_ids.py` をリポジトリ内の JSON ファイルに対して一括実行しました（補助ラッパー `scripts/run_migrate_all.py` を使用）。
- 処理対象: `data/` 以下の JSON ファイル 25 件を走査
- 結果: 7 件成功、18 件失敗（構造がカード列挙ではないファイルや想定外の内部形式を含むため）

成功した主なファイル:
- `data/card_stats.json.migrated`
- `data/card_stats_collected.json.migrated`
- `data/cards.json.migrated`
- `data/scenarios.json.migrated`
- `data/synergy_pairs_v1.json.migrated`
- `data/test_cards.json.migrated`
- `data/test_cards.migrated.json.migrated`

失敗したファイル（手動確認推奨）:
- `data/configs/command_schema.json`
- `data/configs/command_ui.json`
- `data/decks/magic.json`
- `data/decks/緑単3コス.json`
- `data/editor_templates.json`
- `data/examples/*.json` (複数)
- `data/expected_registered_commands.json`
- `data/generations/gen_*.json` (複数)
- `data/locale/ja.json`
- `data/meta_decks.json`
- `data/scenarios/test_query.json`
- `data/sim_settings.json`

注記:
- 失敗の多くはファイルがカード定義の配列ではなく、別用途の JSON (設定・デッキ・ロケール等) であるためです。
- 対応案:
  - カード定義（`cards.json` 等）だけを対象に移行済みであれば当面は問題ありません。
  - 失敗ファイルのうち、カードを含む可能性があるものは個別にレビューして移行を行ってください（例: 一部の `decks/` JSON はカードID列挙のため不要）。

次の推奨作業:
- `data/cards.json` と `data/test_cards.json` の差分を確認し、移行結果をソース管理に追加する（バックアップ推奨）。
- 失敗リストをレビューし、必要なファイルだけを手動で移行またはスクリプトを拡張する。
