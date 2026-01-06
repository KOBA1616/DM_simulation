# 旧Action完全削除ロードマップ (Legacy Action Removal)

## 0. 目的 (Goal)
カードデータ/ツールチェーン内に残る **legacy `Action` 辞書（`actions` フィールド）** と、それを前提にした UI/解析/互換コードを段階的に撤去し、**Command パイプライン（`commands`）を唯一の表現**にする。

- データ表現: `actions` を廃止し `commands` のみ
- 変換: `dm_toolkit.action_to_command.action_to_command` を唯一の入口に固定
- GUI: Action 編集/UI 定義（ActionノードやActionフォーム）を撤去
- 実行: 互換ラッパの縮退・撤去（Command を直接実行する道筋へ）

注: 本ロードマップの「旧Action」は **カードJSONの `actions` および Python 側の legacy Action 辞書・編集UI**を指す。エンジン内部の `core::Action`（プレイヤーインテント/内部処理）は別スコープで、別途の設計決定が必要。

## 1. 現状 (As-Is)
- Editor は Load-Lift で `actions`→`commands` を生成し、通常は `actions` を削除するが、互換フラグにより残存し得る。
- コードベースには `actions` 参照（移行スクリプト、解析、プレビュー、互換経路）が残っている。
- GUI には Action 定義（`ACTION_UI_CONFIG`）が残っており、Action 概念が完全には消えていない。

関連: [docs/command_pipeline_migration.md](../command_pipeline_migration.md)

## 2. フェーズ設計 (Phases)

### Phase 0: 完了条件の固定・観測 (Definition & Gates)
**目的**: 「いつ削除したと言えるか」を明文化し、CIで検出可能にする。

- 完了条件
  - カードデータ（`data/*.json` / `data/cards*.json` 等）に `actions` が存在しない
  - GUI が Action ノード/Actionフォームを表示しない（Action編集導線なし）
  - Python 側の変換入口が `dm_toolkit.action_to_command.action_to_command` に統一
  - `EDITOR_LEGACY_SAVE` などのロールバックフラグを撤去（または no-op 化して最終撤去）

- 検証ゲート（最低限）
  - `python run_pytest_with_pyqt_stub.py -q python/tests/gui/test_gui_stubbing.py`
  - `python run_pytest_with_pyqt_stub.py -q python/tests/dm_toolkit/test_action_migration_coverage.py`

成果物
- 本ドキュメント（完了条件/ゲート）を維持
- CI向けの「`actions` 検出チェック」追加（Phase 2で実装）

---

### Phase 1: 入口の一本化・互換の局所化 (Normalization)
**目的**: 変換ロジックの分散を止め、削除に向けて依存を寄せる。

- 変更内容
  - Action→Command 変換は必ず `dm_toolkit.action_to_command.action_to_command` を経由
  - `dm_toolkit.action_mapper` / GUI の `ActionConverter` の役割を縮退（薄い委譲のみ）
  - `legacy_mode` 的な分岐は `dm_toolkit.compat_wrappers` / `dm_toolkit.unified_execution` に集約

- 完了条件
  - 主要な呼び出し元が `action_to_command` を参照（grep で `map_action` 直叩きが限定される）

---

### Phase 2: データ移行の完了 (Data Migration)
**目的**: リポジトリ内のカードデータから `actions` を根絶し、再導入を防ぐ。

- 変更内容
  - `scripts/python/migrate_actions.py` を公式移行手順として整備（手順を docs に明記）
  - `data/` 配下のカードJSONを一括変換し、`actions` を削除して `commands` のみにする
  - `data/editor_templates.json` などテンプレート/雛形から `actions` を削除
  - CIで `actions` キー混入を検出して失敗させる（簡易スクリプト or pytest）

- 完了条件
  - `data/` から `"actions"` が消える（検索で0件）
  - 新規生成/保存でも `actions` が作られない

---

### Phase 3: GUIから Action 概念を撤去 (GUI Removal)
**目的**: Editorが完全に Commands-only で編集・表示できるようにする。

- 変更内容
  - Actionツリー表示のフォールバック（`effect.get('actions')`）を削除
  - Action UI 定義（`ACTION_UI_CONFIG` / Actionフォーム）を撤去
  - 変数リンク管理など Action依存コードを Command側に寄せる（必要なら `COMMAND_UI_CONFIG` を拡張）

- 完了条件
  - GUIコード内で `ACTION_UI_CONFIG` を参照しない
  - GUIで `actions` ノードが表示されない（フォールバックなし）

---

### Phase 4: 互換スイッチ撤去・実行経路の整理 (Compat Removal)
**目的**: ロールバック/互換を無くして、Commandパイプラインのみで実行できるようにする。

- 変更内容
  - `EDITOR_LEGACY_SAVE` を撤去
  - `DM_ACTION_CONVERTER_NATIVE` など移行中の互換フラグを整理し、最終的に撤去
  - `dm_toolkit.commands.wrap_action` の legacy 依存を縮退（Command オブジェクト/辞書を主経路へ）

- 完了条件
  - Python側で `actions` を扱う互換分岐が残っていない（grepで0に近づける）

---

### Phase 5: デッドコード削除 (Delete)
**目的**: legacy Action 関連モジュール/テスト/ドキュメントを削除して維持コストを0にする。

- 対象例（最終判断）
  - `dm_toolkit/action_mapper.py`（deprecated）
  - `dm_toolkit/gui/editor/action_converter.py`（Command直結に置換後）
  - `dm_toolkit/gui/editor/forms/action_config.py`（Action UI撤去後）
  - `scripts/python/migrate_actions.py`（将来的に不要なら archive へ。ただし再移行需要があるなら残す）

- 完了条件
  - `actions` を前提にしたモジュールが存在しない
  - “旧Action→Command変換”は外部入力（古いデータ）を読む必要がある場合に限って、別途 importable な移行ツールとして隔離

## 3. リスクと対策
- **古いデータの再流入**: CIで `actions` を弾く（Phase 2）。
- **GUI編集互換の喪失**: `legacy_warning` 表示/修正導線を Commands-only で維持。
- **エンジン境界のズレ**: エンジンが期待する enum/型（CommandType/Zone）整合をテストで担保。

## 4. 実行順序（推奨）
- Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

（先にデータだけ消すと、GUIフォールバックが残っていても“動いてしまい”差分が見えにくくなるため、ゲート整備→入口一本化→データ移行→GUI撤去の順が安全）
