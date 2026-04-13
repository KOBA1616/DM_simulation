# 改善TODO（2026-04-13）

## P0（即時対応）

- [x] READMEのドキュメントリンク不整合を修正する
- [x] CI関連ワークフローを削除する（修正ではなく削除方針）
- [ ] status.md の自動更新導線を作成する（手更新の廃止）
- [x] ネイティブ起動前健全性チェックを build 側検証へ移行する
- [x] C++ の `c:\temp\*.txt` 直書きログを撤去して統一ロガーへ寄せる
- [ ] 暫定実装（Temporary/Legacy）の台帳を作成し、期限付きで解消計画を付与する
- [ ] 劣化運転を既定にしない（native 不健全時は失敗させ、根本修正を優先する）

## Native根本修正案（2026-04-13）

- [ ] 実行経路を一本化する（`step` / `fast_forward` / compat の責務分離）
	- [x] `dm_toolkit/commands.py` の暗黙進行（自動 fast_forward）を既定無効化
	- [x] `dm_toolkit/gui/game_session.py` 側で進行責務を一元化
	- [x] human 側 no-action 進行を C++ API (`GameInstance`) へ移管
	- [x] `EngineCompat.ExecuteCommand` 依存経路を縮小し `resolve_command` を正規経路化
	- [x] auto-step worker の停止理由契約（stuck/game_over/waiting_input）を C++ 返却値で明確化
	- [x] 進行経路の契約テストを追加（同一seed同一結果）
- [ ] フェーズ遷移契約を固定する（誰が `next_phase` を呼ぶかを1箇所に限定）
	- [x] `resolve_command_oneshot` の遷移責務を文書化
	- [x] 二重進行を検知するガードテストを追加
- [ ] build時 native 健全性ゲートを標準化する
	- [ ] `build.ps1` の healthcheck をローカル標準手順へ反映
	- [ ] `run_gui.ps1` から起動時探索依存を段階的に縮小
- [ ] fallback を隔離する（`DM_STRICT_NATIVE=1` で契約テスト常時実行）
	- [ ] native契約テストでは Python shim 補正を無効化
	- [ ] fallback 専用テスト群を分離
- [ ] コスト判定と支払い実行のロジックを統合する
	- [ ] `can_pay_cost` と `execute_payment` の入力契約を共通化
	- [ ] ACTIVE_PAYMENT の units 決定ロジックを単一関数へ集約

## P1（構造改善）

- [ ] tests/ と python/tests/ の二重運用を解消し、標準テスト配置を一本化する
- [ ] GUI editor の temporary patch ファイルを本体へ統合する
- [ ] print ベースのログ出力を logger へ統一する
- [ ] stat_key の重複定義を単一ソース化する
- [ ] 障害対応 runbook を症状起点テンプレートへ統一する
- [ ] ルート直下の補助スクリプトを用途別ディレクトリへ整理する

## P2（中長期）

- [ ] ネイティブ回帰の耐障害テストを拡張する（step/fast_forward 周辺）
- [ ] テキスト生成のスナップショット回帰テストを拡張する
- [ ] データ監査ゲートを cards 以外へ拡張する
- [ ] リリース判定基準（品質ゲート）を文書化し、ローカル検証手順へ統合する

## 実施順（推奨）

1. ドキュメント導線固定と status 自動化
2. ネイティブ健全性チェックの build 側移管
3. C++ 一時ログ撤去と暫定実装台帳化
4. テスト配置一本化と stat_key 単一ソース化
5. GUI editor 暫定統合とログ統一

## ユーザー指示（2026-04-13）

- [x] 1. 「同一カードが盤面に出ない」仕様の意図と実装を明文化する
- [x] 2. `c:/temp` 直書きログを全撤去する
- [ ] 3. 劣化運転ではなく根本改善（native 健全性の恒久対応）を優先する
	- [x] `run_gui.ps1` を「不健全 native は失敗（`-AllowFallback` 明示時のみフォールバック）」に変更
	- [x] `build.ps1` 側の健全性チェックを既定化して起動時探索依存を削減
	- [ ] ネイティブ不健全時の根本原因（loader/cost path/phase progression）の恒久修正テストを追加
- [ ] 5. （後回し）ローカル品質ゲートの再整備

## 今回反映済み修正

- [x] README のドキュメントリンクを実在パスへ修正
- [x] GitHub Actions ワークフローを削除（`.github/workflows/` を空に統一）
- [x] README の構造表記の重複項目を整理

## メモ

- 本TODOは、2026-04-13 時点の改善提案を実作業へ落とし込むための管理ファイル。
- 進捗更新時は、関連ドキュメントと実ファイルの整合（リンク切れ/存在確認）を同時に確認すること。
