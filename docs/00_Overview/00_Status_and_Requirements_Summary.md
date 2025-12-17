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

現在、**Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** の最終段階にあり、**Phase 7: Data Migration (全カードデータの新JSONフォーマット移行)** への移行準備を進めています。
`Pure Command Generation` の基盤が整備され、`DrawHandler` などの主要ハンドラーが命令パイプライン形式に移行を開始しました。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` クラスおよびファイルを物理的に削除しました。すべての呼び出し元 (`GameInstance`, `ActionGenerator`, `ScenarioExecutor`, `Bindings`) は `GameLogicSystem` へ移行されました。
*   [Status: Done] **GameLogicSystem Refactor**: アクションディスパッチロジックを `GameLogicSystem` に集約し、`PipelineExecutor` を介した処理フローを確立しました。
*   [Status: Done] **GameCommand**: 新エンジンの核となるコマンドシステム。`Transition`, `Mutate`, `Flow` に加え、`Stat` (統計更新), `GameResult` (勝敗判定) を実装済み。
*   [Status: Done] **Instruction Pipeline**: `PipelineExecutor` が `GAME_ACTION` 命令 (`WIN_GAME`, `LOSE_GAME`, `TRIGGER_CHECK`, `STAT`更新) をサポートするように拡張されました。
*   [Status: WIP] **Pure Command Generation**: `EffectSystem` に `compile_action` メソッドを追加し、アクション定義から `Instruction` リストを生成する仕組みを実装しました。`DrawHandler` の移行が完了（デッキ切れ判定→移動→統計更新→トリガーチェックの命令生成）。
*   [Known Issue] **Binding SegFault**: `Instruction` 構造体の再帰的定義と `nlohmann::json` 引数の Python バインディングにおいて、複雑なオブジェクト受け渡し時に Segmentation Fault が発生する問題が確認されています (`tests/test_effect_compiler.py`)。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Status**: 稼働中 (Ver 2.3)。
*   [Status: Deferred] **Freeze**: 新JSONスキーマが確定次第、新フォーマット専用エディタとして改修を行う。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   [Status: Done] **Status**: パイプライン構築済み。
*   [Status: Blocked] **Pending**: エンジン刷新に伴うデータ構造の変更が確定するまで凍結。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] Phase 6: エンジン刷新 (Engine Overhaul)
[Status: WIP]
`EffectResolver` を解体し、イベント駆動型システムと命令パイプラインへ完全移行します。

*   **Step 1: イベント駆動基盤の実装**
    *   [Status: Done] [Test: Pass]
    *   `TriggerManager`: シングルトン/コンポーネントによるイベント監視・発行システムの実装。（実装完了）
*   **Step 2: 命令パイプライン (Instruction Pipeline) の実装**
    *   [Status: Done] [Test: Pass]
    *   `PipelineExecutor` (VM) を実装済み。
    *   `GAME_ACTION` 命令を追加し、高レベルなゲーム操作（プレイ、攻撃、ブロック）および勝敗判定をパイプライン経由で実行可能にしました。
*   **Step 3: GameCommand への統合**
    *   [Status: Done] [Test: Pass]
    *   `EffectResolver` の主要メソッドを `GameLogicSystem` へ移行。
*   **Step 4: Pure Command Generation (Current Focus)**
    *   [Status: WIP] 各アクションハンドラー (`IActionHandler`) を `compile()` メソッドに対応させ、状態直接操作から命令生成へ移行する。
    *   `DrawHandler`: 完了。
    *   `ManaHandler`: 未着手。
    *   `DestroyHandler`: 未着手。
    *   `SearchHandler`: 未着手。

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

1.  [Status: Todo] **Binding Fix**: `Instruction` 構造体の Python バインディングにおけるメモリ管理・コピーの問題（SegFault）を解決する。
2.  [Status: Todo] **Handler Migration**: `ManaHandler`, `DestroyHandler` 等の主要ハンドラーを `compile()` パターンへ移行する。
3.  [Status: Todo] **EffectResolver Cleanup**: 残存する `EffectResolver` のロジックがあれば完全に `PipelineExecutor` へ統合する。
