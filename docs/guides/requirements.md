# 要件定義（更新） — DM Simulation

## 1. 概要
リポジトリは C++ ネイティブ拡張（`dm_ai_module`）を想定する Python ベースのシミュレータ／学習基盤です。本更新は現行コードベースの挙動とテスト運用を踏まえた短期〜中期の要件整理と、優先改善項目の提示を目的とします。

## 2. 現行主要機能（要約）
- ゲーム状態管理 (`GameState`, `GameInstance`) とデッキ/手札/戦闘ゾーン管理
- コマンド実行および互換レイヤ (`CommandSystem`, `dm_toolkit/engine/compat.py`)
- フェーズ管理 (`PhaseManager`)
- カード定義読み込み（`JsonLoader` / `CardDatabase`）
- AI / 推論ブリッジ（`ParallelRunner`, ネイティブ推論バインディング想定）
- データ収集（`DataCollector`）

## 3. 機能要件（更新）
- インポート/テストの安定性：`import dm_ai_module` が collection 時に失敗しないこと。
- 互換性：ネイティブ未実装のシンボルは Python フォールバックで補い、既存スクリプトとテストが動作すること。
- コマンド実行互換性：`CommandDef`/`Action`/`GameCommand` 等の形状違いに対して寛容に処理できること。
- スタック/エフェクト解決の整合性：pending effects の LIFO 解決、墓地・バトルゾーン等の遷移がテスト期待に一致すること。
- データ収集の整合性：`DataCollector` が呼び出し可能で、最小のインターフェース（`collect_data_batch_heuristic` 等）を提供すること。

## 4. 非機能要件
- テストカバレッジ：既存の pytest 全体がローカルで通ることを短期目標とする。
- 互換性優先：Windows と CI 環境（Linux）で同じ振る舞いが得られるようにする。
- パフォーマンス：当面は Python フォールバック中心のため、性能改善は中期課題とする。

## 5. 制約
- ネイティブ C++ バインディングは現状で部分実装・未ビルドの可能性があるため、恒久解は C++ 側の修正が必要。
- 一部スクリプトはネイティブ API を前提に高速処理を期待している（例：`ParallelRunner`、C++ `DataCollector`）。

## 6. 現在の最大の課題（上位5）
1. ネイティブバインディングの欠落／不整合
   - 症状: collection 時の ImportError / AttributeError。今回の対応で Python フォールバックを追加したが、恒久解はネイティブ実装。
   - 再現箇所: `dm_ai_module` のネイティブロード（`_try_load_native`）／`dm_ai_module.py` でのフォールバック挙動。

2. コマンド/アクション形状の多様性による実行ロジックの脆弱性
   - 症状: `CommandDef`/`Action` の key 名や型（int/str/enum）の違いで実行先が跳ねる。
   - 再現箇所: `dm_toolkit/engine/compat.py` の ExecuteCommand 周り、`CommandSystem.execute_command`。

3. エフェクトスタック（pending_effects）および効果解決の期待差
   - 症状: テストが要求する LIFO 解決や墓地遷移の仕様が未実装/部分実装。
   - 再現箇所: `tests/test_spell_and_stack.py`、`GenericCardSystem.resolve_action`。

4. ネイティブ依存コンポーネントのスタブ不足（DataCollector, ParallelRunner）
   - 症状: スクリプトやテストでこれらを前提としているため存在しないと失敗する。
   - 再現箇所: `training/*`, `tests/test_game_flow_minimal.py`。

5. CI/テストフローの不整備
   - 症状: テスト実行順や環境差で差異が出る可能性。現行はローカルで修正しテストを通したが、自動化が弱い。
   - 再現箇所: `pytest.ini`, ローカル CI スクリプトが存在しない/未整備。

## 7. 短期（1-2日）推奨アクション（優先度順）
1. Python フォールバックの保持と Document 化（短期）
   - 目的: テスト・開発を止めない。今回追加した shim を `dm_ai_module.py` にコメントで明確化し、README に記載。
2. ネイティブバインディングのビルド手順と要修正箇所の特定（短～中期）
   - 目的: 恒久修正のため C++ 側で実装が必要なシンボルを一覧化する。
3. Command/Action 入力の正規化レイヤを拡張（短期）
   - 目的: `compat.py` と `CommandSystem` の変換ルールを充実させる。
4. スタック解決の単体テスト拡充（短期）
   - 目的: pending_effects の期待挙動を明文化し回帰を防ぐ。
5. CI スモークテスト導入（中期）
   - 目的: PR 時に `pytest -q` を実行して重大回帰を防止する。

## 8. 中長期（計画）
- ネイティブバインディングの完全実装とパフォーマンス検証。
- `ParallelRunner` とネイティブ推論ブリッジの高速化とバッチ処理整備。
- テストをカバレッジ指標に紐付けた品質ゲートの導入。

## 9. 受け入れ基準（Acceptance Criteria）
- リポジトリをクローンした新環境で `pytest -q` が成功すること（ローカル CI 追加後は CI 上でも成功）。
- 主要スクリプト（`scripts/` 内）の基本フローが動作すること（簡易サニティチェック）。

## 10. 次の提案タスク（即着手推奨）
- ネイティブ未実装シンボルの一覧化（自動収集スクリプトを追加）
- `dm_toolkit/engine/compat.py` の追加ユニットテスト作成
- README に「Python shim とネイティブの関係」を明記

---

作成者: 自動生成（作業エージェント）
日付: 2026-01-25


