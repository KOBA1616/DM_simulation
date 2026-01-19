# 要件定義 引き継ぎ・実行ガイド

目的: 要件定義書（`01_Requirements_Definition_JP.md`）をプロジェクトメンバーに確実に引き継ぎ、短期〜中期の実行計画を明確にする。

1) 概要
- 参照: `docs/requirements/01_Requirements_Definition_JP.md`
- 本ドキュメントは、後進が理解しやすいように「優先タスク」「検証手順」「受入基準」「連絡先案」を整理したハンドブックである。

2) オンボーディング手順（新人/後進エンジニア向け）
- リポジトリをクローンし、仮想環境を有効化する:

```powershell
git clone <repo>
cd DM_simulation
pwsh -File .\scripts\setup_clone_windows.ps1
& .venv\Scripts\Activate.ps1
```

- 主要ファイルに目を通す（順序）:
  1. `README.md`（プロジェクト概要、セットアップ）
  2. `docs/requirements/01_Requirements_Definition_JP.md`（要件全体）
  3. `docs/DEVELOPMENT_WORKFLOW.md`（開発ルール）
  4. `dm_toolkit/`（Pythonツールキット）

- 最初の動作確認:
  - `python -m pytest -q --maxfail=1` を実行し、環境依存のエラーを把握する。
  - GUI不要テストは `scripts/run_pytest_with_pyqt_stub.py` を参照。

3) 優先タスク（実施手順つき）
- タスクA: `dm_toolkit.gui.i18n` の堅牢化（高）
  - 何をする: Enum 以外のオブジェクトをスキップする防御コードを追加。
  - どう検証する: `python -m pytest -q` が先の AttributeError を超えて進むこと。
  - 所要見積: 0.5〜1日

- タスクB: Action→Command テスト追加（中）
  - 何をする: `tests/` にサンプル Action → 期待 GameCommand を検証するユニットテストを追加。
  - どう検証する: CI とローカル pytest で通過すること。
  - 所要見積: 1〜2週間

- タスクC: ドキュメント整理とアーカイブ運用（中）
  - 何をする: `docs/` の古いメモは `archive/docs/` に移し、`docs/` は現行参照のみを残す運用を確立。
  - どう検証する: `docs/` にあるファイルが参照ドキュメントとして完結すること。

4) 受入基準（Hand-off Acceptance）
- 新任者が次のコマンドで主要テストを実行し、致命的な例外が発生しないこと:

```powershell
& .venv\Scripts\Activate.ps1
python -m pytest -q --maxfail=1
```

- 優先タスクA の修正がマージされていること（PR のリンクを添付）
- `README.md` のセットアップ手順が正常に動作すること

5) コミュニケーション & 受け渡しチェックリスト
- PR を作成する際は以下を含める:
  - 目的と範囲
  - 再現手順（テストコマンド）
  - 受入基準（pass基準）
  - 影響範囲（変更するファイルと影響するモジュール）

- 受け渡し時に行う簡易レビュー項目:
  - ローカル pytest の実行結果のスクリーンショット/ログ
  - 主要モジュールの簡単な説明（5行程度）
  - 既知の未解決課題と優先度

6) 追加資産（推奨テンプレート）
- PR テンプレート（短い説明/検証手順/受入基準）を `.github/PULL_REQUEST_TEMPLATE.md` に追加することを推奨。

---

このファイルは後進への引き継ぎ用です。内容を追加・修正したい場合は PR を作成してください。
