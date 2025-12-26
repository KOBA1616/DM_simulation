Migration Canary — テレメトリ実行手順

目的: ステージ環境で `Action`→`Command` 自動変換の挙動を観測し、変換不能ケースの発生率を収集する。

1. ステージ環境（またはローカル）でリポジトリルートに移動。
2. カナリースクリプトを実行（PowerShell推奨）:

```powershell
.\scripts\run_migration_canary.ps1
```

3. 実行後、`migration_metrics_canary.jsonl` に変換イベントが JSONL 形式で追記されます。
   - 各行は `{ ts, success, action_type, warning }` の JSON です。
4. ログ集計例（PowerShell）:

```powershell
Get-Content migration_metrics_canary.jsonl | ConvertFrom-Json | Group-Object -Property success | Select-Object Count, Name
```

5. 次のアクション:
   - `warning` が多い `action_type` を抽出して優先修正リストを作成
   - GUI上での `WarningCommand` 編集体験を改善

注意: 実稼働データ収集時はプライバシーと合意を確認してください（匿名化 / オプトイン推奨）。
