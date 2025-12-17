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

現在、**Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** が完了し、**Phase 7: Data Migration (全カードデータの新JSONフォーマット移行)** への移行フェーズにあります。
`EffectResolver` の解体と `GameCommand` / `PipelineExecutor` への移行が完了し、イベント駆動型アーキテクチャへの基盤移行が達成されました。

AI学習 (Phase 3) およびエディタ開発 (Phase 5) は、このエンジン刷新が完了するまで一時凍結します。

### 重要変更 (Strategic Shift)
既存のJSONデータ（Legacy JSON）の再利用や変換アダプタ (`LegacyJsonAdapter`) の開発は**完全に放棄・廃止**します。
今後は新エンジン (`CommandSystem` / `PipelineExecutor`) に最適化された新しいJSON形式のみをサポートし、過去の資産に縛られず、エンジンの完成度と品質を最優先します。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` クラスおよびファイルを物理的に削除しました。すべての呼び出し元 (`GameInstance`, `ActionGenerator`, `ScenarioExecutor`, `Bindings`) は `GameLogicSystem` へ移行されました。
*   [Status: Done] **GameLogicSystem Refactor**: アクションディスパッチロジックを `GameLogicSystem` に集約し、`PipelineExecutor` を介した処理フローを確立しました。
*   [Status: Done] **GameCommand**: 新エンジンの核となるコマンドシステム。`Transition`, `Mutate`, `Flow` などのプリミティブを実装済み。
*   [Status: Done] **Instruction Pipeline**: `PipelineExecutor` が `GAME_ACTION` 命令を介して高レベルなゲームロジックを実行する仕組みが確立しました。
*   [Status: Fixed] **Binding SegFault**: `GameInstance` の Python バインディングにおいて発生していた Segmentation Fault (特に `initialize_card_stats` や `resolve_action` 呼び出し時) に対処しました。
    *   原因: Python辞書として渡された `card_db` が Pybind11 により巨大な `std::map` へ毎回変換・コピーされ、スタックオーバーフローまたはメモリ破損を引き起こしていた可能性が高い。
    *   対策: `GameInstance` に `resolve_action` と `initialize_card_stats` メソッドを実装し、Python側からデータを渡さずに C++ 内部で保持している `card_db` (shared_ptr) を利用するようにアーキテクチャを変更しました。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Status**: 稼働中 (Ver 2.3)。
*   [Status: Deferred] **Freeze**: 新JSONスキーマが確定次第、新フォーマット専用エディタとして改修を行う。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   [Status: Done] **Status**: パイプライン構築済み。
*   [Status: Blocked] **Pending**: エンジン刷新に伴うデータ構造の変更が確定するまで凍結。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] Phase 6: エンジン刷新 (Engine Overhaul)
[Status: Done]
`EffectResolver` を解体し、イベント駆動型システムと命令パイプラインへ完全移行しました。

*   **Step 1: イベント駆動基盤の実装**
    *   [Status: Done] [Test: Pass]
    *   `TriggerManager`: シングルトン/コンポーネントによるイベント監視・発行システムの実装。（実装完了）
    *   `check_triggers` メソッドにより、`GameEvent` をトリガーとして `PendingEffect` を生成するフローを確立。
*   **Step 2: 命令パイプライン (Instruction Pipeline) の実装**
    *   [Status: Done] [Test: Pass]
    *   `PipelineExecutor` (VM) を実装済み。
    *   `GAME_ACTION` 命令を追加し、高レベルなゲーム操作（プレイ、攻撃、ブロック）をパイプライン経由で実行可能にしました。
    *   [Status: Done] **Deleted**: `LegacyJsonAdapter` は廃止されました。
*   **Step 3: GameCommand への統合**
    *   [Status: Done] [Test: Pass]
    *   `EffectResolver` の主要メソッド（`resolve_play_card`, `resolve_attack` 等）を `GameLogicSystem` へ移行し、内部処理を全て `GameCommand` (Transition, Mutate) で書き換えました。
    *   [Status: Done] **New**: `GameInstance` にて `TriggerManager` を `GameState::event_dispatcher` と連携させ、コマンド実行時のイベント発行をトリガー検知につなげる統合を完了。
*   **Step 4: EffectResolver 完全撤廃 (Final Cleanup)**
    *   [Status: Done] `src/engine/effects/effect_resolver.*` を削除。
    *   [Status: Done] Python Bindings から `EffectResolver` を削除し、`GameLogicSystem` へ移行。
    *   [Status: Done] `GameInstance` を更新し、`card_db` のライフタイム管理を改善 (`shared_ptr` 導入)。

### 3.2 [Priority: High] Phase 7: ハイブリッド・エンジン基盤 & データ移行
[Status: WIP]
旧エンジン（マクロ的アクション）と新エンジン（プリミティブコマンド）を共存・統合させ、全てのデータを新形式へ移行します。

*   **Step 1: データ構造の刷新 (Hybrid Schema)**
    *   [Status: Done] [Test: Pass]
    *   JSONスキーマに `CommandDef` を導入済み。
    *   `GenericCardSystem::resolve_effect` を更新し、旧来の `actions` と共に新しい `commands` を反復処理・実行するロジックを実装しました。
*   **Step 2: CommandSystem の実装**
    *   [Status: Done] [Test: Pass]
    *   `dm::engine::systems::CommandSystem` を実装。
    *   `MUTATE` 処理（TAP, POWER_MOD, ADD_KEYWORD等）の実装を完了し、Pythonテスト `test_command_system.py` にて動作検証済み。
*   **Step 3: フロー制御 (Control Flow) の実装**
    *   [Status: Done] [Test: Pass]
    *   `CommandDef` に条件分岐用の `condition`, `if_true`, `if_false` フィールドを追加 (Hybrid Schema拡張)。
    *   `test_command_system.py` にて `FLOW` コマンド（条件合致時の分岐、不一致時の分岐）の動作検証を完了しました。
*   **Step 4: Python Binding 修正 (Resolved)**
    *   [Status: Done] **Fix SegFault**: `GameInstance` に `resolve_action` と `initialize_card_stats` を実装・バインドし、巨大な `card_db` オブジェクトの受け渡しを廃止することでクラッシュを解消しました。

### 3.3 [Priority: Future] Phase 8: Transformer拡張 (Hybrid Embedding)
[Status: Deferred]
Transformer方式を高速化し、かつZero-shot（未知のカードへの対応）を可能にするため、「ハイブリッド埋め込み (Hybrid Embedding)」を導入します。また、文脈として墓地のカードも対象に含めます。

*   **コンセプト (Concept)**
    *   **Hybrid Embedding**: `Embedding = Embed(CardID) + Linear(ManualFeatures)`
    *   **Zero-shot対応**: 未知のカード（ID埋め込み未学習）でも、スペック情報（コスト、文明、パワー等）から挙動を推論可能にする。
    *   **スコープ拡張**: Transformerの入力文脈に「墓地」のカードも含め、墓地利用や探索に対応させる。

*   **実装要件 (Requirements)**
    *   [Status: Todo] **A. C++側 (TensorConverter)**
        *   `convert_to_sequence` を修正し、`Output: (TokenSequence, FeatureSequence)` を返すように変更する。
    *   [Status: Todo] **B. Python側 (NetworkV2)**
        *   モデル入力層を修正: `x_id` (Card IDs) と `x_feat` (Manual Features) を受け取る。
    *   [Status: Todo] **C. 特徴量ベクトルの定義 (Feature Vector Definition)**
        *   **Card Function Embeddings (機能抽象化)**: 除去、ドロー、ブロッカー等の役割をベクトル化し、未知のカードに対応（Zero-shot）。
        *   **Projected Effective Mana (次ターン有効マナ)**: 次ターンのアンタップマナ数と発生可能文明を予測し、色事故を防ぐ。
        *   **Synergy/Combo Readiness (コンボ成立度)**: 進化元やコンボパーツの揃い具合を数値化し、戦略的キープを促進。
        *   **Turns to Lethal (リーサル距離)**: LethalSolverによる「勝利/敗北までのターン数」を入力し、終盤の判断力を強化。
        *   その他: 基本スペック（コスト、パワー、文明）、キーワード能力、リソース操作等。
    *   [Status: Todo] **D. Advanced Lethal Search (Mini-Max)**
        *   現行のGreedy Heuristicに加え、手札（SA、進化速攻）やトリガーリスクを考慮した「超短期探索型 (Mini-Max Search) LethalSolver」を実装する。
        *   エンジンをコピーしてシミュレーションを行うことで、攻撃時効果やブロッカー除去も正確に判定可能にする。

## 4. 今後の課題 (Future Tasks)

1.  [Status: Todo] **Phase 7: Data Migration**:
    *   既存のJSONカードデータ (`actions` ベース) を新しい `commands` ベースのフォーマットへ完全に移行する。
2.  [Status: Todo] **New JSON Standard Adoption**:
    *   Legacyサポート完全撤廃に伴い、全データセットの再構築を行う。
3.  [Status: Todo] **GUI Update**:
    *   `CardEditor` を更新し、新スキーマ (`CommandDef`) の編集に対応させる。
