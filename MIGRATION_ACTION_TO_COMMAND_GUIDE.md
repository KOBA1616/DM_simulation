## 概要（日本語要約）

- ステータス: コマンド優先（Command-first）方式への移行を段階的に実施中。主要な Python 側互換層とトレーニング／ツール群はコマンド優先へ切替済みで、ローカル回帰テストは合格しています（最終ローカル実行: 69 passed, 4 skipped）。
- 次のステップ: 現状の変更を PR にまとめて CI 上でフルスイートを実行し、パリティを確認したうえで古い Action 系 shim を段階的に削除します。

このドキュメントは「Action ベース → Command ベース」への安全な移行手順、実施記録、残タスク、マージ前チェックリストを日本語でまとめたものです。C++ が最終的な真実のソース（source of truth）であり、Python は薄い互換レイヤーに留める方針です。

---

## 主要な判断と方針

- C++ を生成・実行の単一ソースとする。Python はログ／テスト／UI 用の薄いアダプタに限定する。
- 生成器は可能な限り CommandDef（コマンド定義）を返すようにし、Action は互換レイヤー経由の一時的フォールバックに留める。
- 変更は小さなバッチで行い、各バッチ後に代表テスト + フルテストを実行して回帰がないことを確認する。

---

## これまでの主要作業（ハイライト）

- 中央互換ヘルパーを実装: `dm_toolkit/training/command_compat.py`（コマンド優先で生成・正規化・エンコーディングを行う中央ヘルパー）。
- トレーニング・ツール群を段階的に更新: `training/head2head.py`, `training/fine_tune_with_mask.py`, `training/ai_player.py`, `tools/emit_play_attack_states.py` 等をコマンド優先に変更し、互換ヘルパーで Action/Command の正規化を行う実装に置換。
- データ検証を追加: 訓練データの policy ベクトル長を `dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE` と突き合わせる保護を訓練スクリプトに追加。
- DataCollector 周りの正規化: `dm_toolkit/training/collect_training_data.py` がネイティブから受け取るコマンド様オブジェクトを canonical な確率ベクトルに変換する処理を導入。
- ツール類の更新: `tools/check_policy_on_states.py` 等が command-first の生成器を優先して利用するように変更。
- ドキュメントと todo 管理を更新し、ローカルでフルテストを複数回実行して回帰を修正・確認（最終: 69 passed, 4 skipped, 13 warnings）。

---

## 実装フェーズ（短縮版）

1. フェーズ0 — 現状把握（完了）: Action/Command 呼び出し箇所を網羅。
2. フェーズ1 — C++ に Command ジェネレータを追加（完了）。
3. フェーズ2 — pybind で Command-first API を公開（完了）。
4. フェーズ3 — Python 側コマンド優先モジュール導入（完了）。
5. フェーズ4 — Python 実行パスを Command-first に移行（進行中、トレーニング/ツールは主要箇所を変換済み）。
6. フェーズ5 — 古い shim の削除と最終化（CI パリティ確認後に段階実行）。

---

## 主要ファイル（簡潔）

- `dm_toolkit/training/command_compat.py` — 中央互換ヘルパー（generate_legal_commands, normalize_to_command, command_to_index）。
- `training/head2head.py`, `training/fine_tune_with_mask.py`, `training/ai_player.py` — トレーニングループや推論クライアントでコマンド優先化。
- `dm_toolkit/training/collect_training_data.py` — DataCollector の出力がコマンド様オブジェクトだった場合に canonical ポリシーへ変換する処理を追加。
- `tools/check_policy_on_states.py`, `tools/emit_play_attack_states.py` — コマンド優先の生成器を利用。

---

## テストと検証状況

- ローカルフルテスト: 69 passed, 4 skipped, 13 warnings（回帰なし）。
- 代表的な parity テストは都度実行し、問題点が出た箇所は互換ラッパーを修正して対応。

---

## リスクと対策

- データ次元ミスマッチ: 古いデータセットのポリシー長が新しい `CommandEncoder.TOTAL_COMMAND_SIZE` と一致しない場合、`training/convert_training_policies.py` で変換する。訓練スクリプト側で警告/停止する保護を追加済み。
- テスト回帰: 各バッチで代表テスト＋フルテストを実行。問題があれば互換ラッパーを拡張して暫定対応。
- ランタイム互換性（C++/pybind）: C++ 側の API が変更されると破壊的になるため、pybind の互換層は慎重に扱う。

---

## 残タスク（優先度）

1. CI パリティ実行（高）: PR を作成して CI 上でフルスイートを通す。CI が緑であれば削除作業に進む。
2. テスト群の段階的修正（中）: 一部テストが古い Action 出力を前提としているため、Command 前提に書き換える（小分けで実施）。
3. `ActionGenerator` shim の段階的削除（高）: CI パリティ確認後、影響の小さい箇所から削る。
4. `CommandDef.to_dict()` の C++ 実装と公開（中）: ログ／Telemetry の整合性を確保。
5. 長期安定性試験（低）: 自己対戦や長時間負荷試験で安定性を確認。

---

## マージ前チェックリスト

- [ ] PR 作成 → CI フルスイートを実行し全通過を確認する。
- [ ] 代表的な end-to-end スクリプト（`training/head2head.py` 等）の短時間実行で動作確認する。
- [ ] リポジトリ内に残る `ActionGenerator` 参照を特定し、段階的に削除計画を立てる。
- [ ] データセットの policy ベクトル長を `CommandEncoder.TOTAL_COMMAND_SIZE` と一致させる（必要なら `training/convert_training_policies.py` を使用）。
- [ ] 重要ログが `CommandDef.to_dict()` を用いることを確認する。

---

## 推奨次アクション（選択肢）

- A: この状態で PR を作成して CI を回す（非破壊、推奨）。
- B: 残りの training ファイルを 5–10 ファイルずつバッチで変換し、各バッチでテストする（慎重）。
- C: 先にテスト群を Command 前提に書き換える（破壊的、段階的に実施推奨）。

どれを実行するか指示ください。選択に応じて PR 作成、または次バッチの変換を進めます。
