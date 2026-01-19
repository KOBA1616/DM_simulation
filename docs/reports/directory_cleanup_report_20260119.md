```markdown
# Directory cleanup report — 2026-01-19

## 実施内容（要約）
- `AGENTS.md` をレビューし、ドキュメント参照の不整合（`action_mapper`の記載）を注記で修正。
- 要件レビュー報告を `docs/requirements/requirements_review_20260119.md` として作成。
- 実装メモ（実験/機能実装記録）を `archive/docs/` に移動（コピー）し、ルートの元ファイルを簡略版に差し替え：
  - `MEGA_LAST_BURST_IMPLEMENTATION.md` → `archive/docs/MEGA_LAST_BURST_IMPLEMENTATION.md`
  - `CAST_SPELL_REPLACE_CARD_MOVE_IMPLEMENTATION.md` → `archive/docs/CAST_SPELL_REPLACE_CARD_MOVE_IMPLEMENTATION.md`
  - `IF_CONDITION_LABELS.md` → `archive/docs/IF_CONDITION_LABELS.md`
- GUIヘッドレススタブ関連のテストをヘッドレス実行で検証：
  - 実行コマンド: `python scripts/run_pytest_with_pyqt_stub.py python/tests/gui/test_gui_stubbing.py -q`
  - 結果: 1 passed

## 変更ファイル（主なもの）
- [AGENTS.md](../Specs/AGENTS.md)
- [docs/requirements/requirements_review_20260119.md](../requirements/requirements_review_20260119.md)
- [./directory_cleanup_report_20260119.md](./directory_cleanup_report_20260119.md) (本報告)
- アーカイブ先（コピー済）:
  - [archive/docs/MEGA_LAST_BURST_IMPLEMENTATION.md](../../archive/docs/MEGA_LAST_BURST_IMPLEMENTATION.md)
  - [archive/docs/CAST_SPELL_REPLACE_CARD_MOVE_IMPLEMENTATION.md](../../archive/docs/CAST_SPELL_REPLACE_CARD_MOVE_IMPLEMENTATION.md)
  - [archive/docs/IF_CONDITION_LABELS.md](../../archive/docs/IF_CONDITION_LABELS.md)

## 推奨の次ステップ
1. CI上でフルテスト実行（ヘッドレス GUI スタブ有効）: `python scripts/run_pytest_with_pyqt_stub.py -q` を CI ジョブに追加。
2. ドキュメント整理のポリシーを README または `docs/00_Overview/DEVELOPMENT_WORKFLOW.md` に記載して、今後の作業でルート直下に実装メモを作成しない運用にする。
3. アーカイブ方針の承認後、ルートに残る簡略参照ファイルを削除または `.archive` プレフィックス付与して明示する（必要なら私が実行します）。

## 補足
- 実装に影響するソース変更は行っていません。すべての編集はドキュメント整理（非破壊：詳細は archive に保存）に限ります。

---

作業者: 自動整備エージェント
作業日: 2026-01-19

```
