# Windows Native Crash Collection Guide

目的: `pytest` 実行中のネイティブ異常終了（例: `0xC0000409` / `3221226505`）を、再現時に同一手順で採取する。

## 1. 前提

- `tools/procdump64.exe` が存在すること
- `.venv` が作成済みであること

## 2. 基本コマンド

```powershell
pwsh -File scripts/run_with_procdump.ps1 -PytestArgs @('-m','pytest','tests/','-q') -DumpDir dumps/native_crash -CollectEventLog
```

出力:
- dump ファイル: `dumps/native_crash/*.dmp`
- 実行メタ情報: `dumps/native_crash/run_with_procdump_meta.txt`
- Event Log 抜粋: `dumps/native_crash/windows_application_errors.txt`

## 3. 重点再現（シャッフル実行）

```powershell
pwsh -File scripts/run_with_procdump.ps1 -PytestArgs @('scripts/run_tests_shuffled.py','-n','20','--log-dir','logs/shuffled2') -DumpDir dumps/shuffled_crash -CollectEventLog
```

補足:
- `run_tests_shuffled.py` を Python スクリプトとして直接実行するため、`-PytestArgs` は実際には Python 引数列として扱う。

## 4. 収集後に共有すべきファイル

- `dumps/*/*.dmp`
- `dumps/*/run_with_procdump_meta.txt`
- `dumps/*/windows_application_errors.txt`
- `logs/shuffled2/*.log`（シャッフル時）
- `logs/native_repeat/*.txt`（repeat 実行時）

## 5. トラブルシュート

- `ProcDump executable not found`:
  - `tools/procdump64.exe` の配置を確認する。
- dump が生成されない:
  - 例外が未発生か、対象プロセスが短時間で終了している可能性がある。
  - `-PytestArgs` を最小再現ケースに絞って再試行する。
- Event Log が空:
  - 管理者権限や発生時刻のズレを確認し、再試行する。
