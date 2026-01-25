ヘッドレス実行（GUIなし）手順

- 前提: 仮想環境を有効化しておく（プロジェクトルートに `.venv` がある想定）

PowerShell での実行例:

```powershell
# 仮想環境を有効化
& .\.venv\Scripts\Activate.ps1

# ビルドしてからヘッドレス実行（オプション）
.\scripts\run_headless.ps1 -Build

# 追加引数は headless スクリプトへそのまま渡されます（例: --p0-human, --p1-human, --steps 50）
.\scripts\run_headless.ps1 -- --p0-human human --steps 50
```

- 目的: CI やローカルで GUI を起動せずにゲーム進行ロジックを検証するためのラッパーです。
- 参照スクリプト: `scripts/headless_mainwindow.py`, `scripts/headless_smoke.py`。
- 既存テスト: `tests/test_headless_smoke.py` がヘッドレス実行のサンプルです。