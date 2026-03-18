# SELECT 回帰監視 Runbook

目的: `SELECT -> TRANSITION` 系の順序依存や再発を、同一手順で継続監視する。

## 1. 実行コマンド

```powershell
c:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe scripts/run_select_regression.py -n 5 --log-dir logs/select_regression
```

出力:
- 反復ログ: `logs/select_regression/select_run_<N>.log`
- サマリ: `logs/select_regression/summary.json`

## 2. 監視対象テスト

- `tests/test_transition_reproducer.py`
- `tests/test_card1_hand_quality.py`

## 3. 判定基準

- PASS: `summary.json` の `results[*].ok` がすべて `true`
- FAIL: いずれかの `ok` が `false`

## 4. 失敗時手順

1. `logs/select_regression/summary.json` の `failed_test` を確認する。
2. 該当反復ログ (`log_path`) を開き、最初の失敗スタックを確認する。
3. 失敗テストのみを単体再実行する。

```powershell
c:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe -m pytest <failed_test_path> -q --maxfail=1
```

4. 失敗が再現する場合は、関連ログを保存して報告する。
- `logs/select_regression/select_run_<N>.log`
- `logs/pipeline_trace.txt`（存在する場合）
- `logs/transition_debug.txt`（存在する場合）

## 5. 運用メモ

- CI ナイトリーで実行する場合も同じコマンドを使い、`summary.json` を成果物として保存する。
- 手動監視では `-n 3` 以上を推奨する。
