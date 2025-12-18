# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## ステータス定義 (Status Definitions)
開発状況を追跡するため、各項目の先頭に以下のタグを付与してください。

*   `[Status: Todo]` : 未着手。AIが着手可能。
*   `[Status: WIP]` : (Work In Progress) 現在作業中。
*   `[Status: Review]` : 実装完了、人間のレビューまたはテスト待ち。
*   `[Status: Done]` : 完了・マージ済み。
*   `[Status: Blocked]` : 技術的課題や依存関係により停止中。
*   `[Status: Deferred]` : 次期フェーズへ延期
*   `[Test: Pending]` : テスト未作成。
*   `[Test: Pass]` : 検証済み。
*   `[Test: Fail]` : テスト失敗。修正が必要。
*   `[Test: Skip]` : 特定の理由でテストを除外中。
*   `[Known Issue]` : バグを認識した上でマージした項目。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

現在、**Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** がほぼ完了し、仕上げの段階に入っています。
`PipelineExecutor` による動的な効果解決、S・トリガーのスクリプト実行、およびループ内での一時停止・再開が実装され、バックエンド側の基盤整備が完了しました。

直近の更新で、`TransitionCommand`, `MutateCommand`, `AttachCommand` などの主要なコマンド実装が完了し、`GameLogicSystem` によるアクションディスパッチがパイプライン経由で行われるようになりました。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` クラスおよびファイルを物理的に削除し、ロジックを `GameLogicSystem` と `PipelineExecutor` に完全移行しました。
*   [Status: Done] **GameLogicSystem Refactor**: アクションディスパッチロジックを `GameLogicSystem` に集約し、`PipelineExecutor` を介した処理フローを確立しました。
*   [Status: Done] **GameCommand**: `Transition`, `Mutate`, `Flow`, `Stat`, `GameResult`, `Attach` などの基本コマンド群を実装済み。
*   [Status: Done] **Instruction Pipeline**: `PipelineExecutor` が動的インストラクション注入 (`inject_instructions`) と、ループ状態の保存・復元 (`ExecutionFrame`, `LoopState`) をサポート。
*   [Status: Done] **S-Trigger & Evolution**: S・トリガーの動的注入、進化クリーチャーの重ね合わせ処理 (`AttachCommand`) を実装済み。
*   [Status: Review] **Loop Resume Support**: パイプラインの一時停止・再開機能を実装。ループ (`LoopState`) の途中状態からの復帰が可能となりました（検証用スクリプト作成済み）。
*   [Status: WIP] **Pure Command Generation**: `IActionHandler` の `compile` メソッド移行を進めており、主要なアクション（ドロー、ディスカード、破壊、プレイ）は対応済み。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Status**: 稼働中 (Ver 2.3)。
*   [Status: Deferred] **Freeze**: 新JSONスキーマが確定次第、新フォーマット専用エディタとして改修を行う。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   [Status: Done] **Status**: パイプライン構築済み。
*   [Status: Blocked] **Pending**: エンジン刷新に伴うデータ構造の変更が確定するまで凍結。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] Phase 6: エンジン刷新 (Engine Overhaul)
[Status: Review]
エンジンコアの再設計は完了し、最終的な統合テストとUI連携待ちです。

*   **Step 1: イベント駆動基盤の実装**
    *   [Status: Done] [Test: Pass]
    *   `TriggerManager`: イベント監視・発行システム実装済み。
*   **Step 2: 命令パイプライン (Instruction Pipeline) の実装**
    *   [Status: Done] [Test: Pass]
    *   `PipelineExecutor` 実装済み。
*   **Step 3: GameCommand への統合**
    *   [Status: Done] [Test: Pass]
    *   `GameLogicSystem` への移行完了。
*   **Step 4: Pure Command Generation & Integration**
    *   [Status: Review] **Task A: Complete Effect Resolution**: S・トリガー等の効果をパイプラインで実行する仕組みを実装。
    *   [Status: WIP] **Task B: Refined Evolution Filters**: 進化条件（種族・文明）の詳細チェックロジックの実装（次フェーズ着手）。
    *   [Status: Review] **Task C: Loop Resume Support**: ループ内での一時停止・再開ロジックの実装完了。
    *   [Status: Todo] **Task D: Frontend Integration**: GUI側での入力待ち状態監視とダイアログ表示の実装。

### 3.2 [Priority: High] Phase 7: ハイブリッド・エンジン基盤 & データ移行
[Status: Pending]
全てのデータを新形式へ移行します。

*   **Step 1: データ構造の刷新 (Hybrid Schema)**
    *   [Status: Done] [Test: Pass]
    *   JSONスキーマに `CommandDef` を導入済み。
*   **Step 2: CommandSystem の実装**
    *   [Status: Done] [Test: Pass]
    *   `dm::engine::systems::CommandSystem` を実装。

## 4. 今後の課題 (Future Tasks)

1.  [Status: Todo] **Frontend Integration**: `GameInstance::is_waiting_for_input()` を監視し、Python GUI (`app.py`) 側で適切なユーザーインターフェースを表示して `resume_processing` を呼び出す実装。
2.  [Status: Todo] **Refined Evolution Filters**: 進化条件（種族・文明・コスト）の厳密なフィルタリングロジックの実装。
3.  [Status: Todo] **Full Handler Migration**: 残りのアクションハンドラーを `compile()` パターンへ移行し、レガシーな `resolve` ロジックを完全に排除する。
4.  [Status: Todo] **Integration Verification**: 実際のゲームシナリオを用いたE2Eテストを行い、エンジンの動作を保証する。
