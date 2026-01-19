# Requirements review — 2026-01-19

## 概要
本レビューはリポジトリ内の要件定義と現在の実装（軽い走査）を照合し、更新・整理・アーカイブの候補を提示します。主要参照は `AGENTS.md`。

## 主な所見
- `AGENTS.md` に記載された方針は明確で、特に「コマンド正規化（`dm_toolkit.action_to_command.action_to_command`）」の一元化方針は正しい方向性。
- ヘッドレステスト、スタブ注入、`dm_ai_module` のロード優先度に関する運用ルールが定義されている。
- ドキュメント内に挙がるフェーズ（Phase 6: QA）が未完であり、GUIスタブ関連のテスト失敗が残っているとの記述あり。

## 推奨アクション（優先順）
1. 実装差分チェック
   - 対象: `dm_toolkit/` 以下、`dm_ai_module.py`、`conftest.py`、`run_pytest_with_pyqt_stub.py`。
   - 目的: ドキュメント通りに `action_to_command` が唯一の正規化経路になっているか、互換ラッパーの責務分離が守られているかを確認。
2. GUIスタブ失敗ログの収集と分類
   - 対象: `python/tests/gui/test_gui_stubbing.py` と直近のCI/ローカルログ。
   - 目的: 高優先度の不具合を特定し、Phase 6の完了基準を定義する。
3. ドキュメント整理（非破壊）
   - 既存の方針文書は `docs/00_Overview/` 以下に要約を移し、個別実装メモ（過去の実験的メモ等）は `archive/` に移動推奨。
4. アーカイブ実行（運用合意後）
   - 過去の実験ノートや重複する実装メモを `archive/docs/` に移動して参照用に残す。

## アーカイブ候補（要確認）
- `MEGA_LAST_BURST_IMPLEMENTATION.md`
- `CAST_SPELL_REPLACE_CARD_MOVE_IMPLEMENTATION.md`
- （古い実験ノートや重複ファイルは手動で査定を推奨）

## 次ステップ提案
- 私からの実行案: まず `dm_toolkit` と `dm_ai_module.py` を走査して差分レポート（短いパッチ候補）を作成します。続けてGUIスタブの直近テストログを収集します。これらを実施してよいですか？

---

*作成日: 2026-01-19*
