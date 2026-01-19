# 要件サマリーとトレーサビリティ

目的: `01_Requirements_Definition_JP.md` に記載された要件を短く参照できる形でまとめ、担当・優先度・受入基準への参照を明示する。

1) 要件一覧（抜粋）
- FR-1: ゲームルール実行 — 実装箇所: `src/engine/`, 責任者案: `engine-team`、優先度: 高、受入基準: 単体テストと統合テストで合格
- FR-2: 行動表現の標準化（Action→Command） — 実装箇所: `dm_toolkit/action_to_command.py`、責任者案: `ai-team`、優先度: 高
- FR-3: 互換性（レガシーAPI） — 実装箇所: `dm_toolkit/compat_wrappers.py`、責任者案: `integration-team`、優先度: 中
- FR-4: AI学習パイプライン — 実装箇所: `training/`, `dm_toolkit/ai/`、責任者案: `ml-team`、優先度: 高
- FR-5: GUIツール（カードエディタ） — 実装箇所: `dm_toolkit/gui/`、責任者案: `gui-team`、優先度: 中
- FR-6: テストと検証（Headless含む） — 実装箇所: `scripts/`, `tests/`、責任者案: `qa-team`、優先度: 高

2) トレーサビリティマトリクス（簡易）
- `FR-1` → `src/engine/*`, `tests/engine_test.py`
- `FR-2` → `dm_toolkit/action_to_command.py`, `tests/dm_toolkit/test_action_to_command.py`
- `FR-4` → `training/*`, `scripts/generate_training_data.py`, `models/`
- `FR-5` → `dm_toolkit/gui/editor/`, `docs/GUI_HEADLESS_TESTING_SETUP.md`

3) 優先度別アクション（短期）
- 高: FR-1, FR-2, FR-4, FR-6 — 直近でCIとpytestが通る状態を作る
- 中: FR-3, FR-5 — 互換ラッパーとGUIの安定化

4) 受入基準へのリンク（チェック用コマンド）
- 単体テスト: `python -m pytest tests/ -q`
- 全体テスト（推奨短時間確認）: `python -m pytest -q --maxfail=1`
- ドキュメント確認: `docs/requirements/01_Requirements_Definition_JP.md` を参照

5) 推奨レビュー/承認フロー
- 小さな修正（1つのモジュール）: 1 PR、1 レビュワー、マージ後 CI 通過で OK
- 仕様変更や設計修正: RFC スタイルの PR（説明・影響・移行手順を明記）、2 レビュワー以上

6) 次のアクション（今週）
- A1: `dm_toolkit.gui.i18n` の防御的修正（優先 — テスト阻害要因）
- A2: `tests/dm_toolkit/test_action_to_command.py` の整備（カバレッジ確認）
- A3: CI ワークフローに `pytest --maxfail=1` を追加（短期の自動検出）

---

この要件サマリーは軽量で頻繁に参照されることを想定しています。詳細は `01_Requirements_Definition_JP.md` を参照してください。
