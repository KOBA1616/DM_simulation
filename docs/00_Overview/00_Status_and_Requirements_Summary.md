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

現在、**Phase 1: Game Engine Reliability** および **Phase 6: Engine Overhaul** の実装が完了し、C++コア（`dm_ai_module`）のビルドは安定しています。
2025年2月の開発サイクルにおいて、コスト軽減システムの柔軟性向上に取り組み、動的な値（ドロー枚数など）に基づくコスト計算を実現しました。

直近では「動的コスト軽減（Dynamic Cost Reduction）」機能を実装し、特定の統計値（カードドロー数、マナ数、シールド数など）を参照してコストを増減させるメカニズムを導入しました。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **Card Owner Refactor**: `CardInstance` 構造体に `owner` フィールドを追加し、外部マップ `card_owner_map` を廃止しました。
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` を削除し、`GameLogicSystem` へ完全移行しました。
*   [Status: Done] **GameLogicSystem Refactor**: `PipelineExecutor` を介したコマンド実行フローが確立されました。
*   [Status: Done] **Action Generalization**: 全アクションハンドラーの `compile_action` 化が完了しました。
*   [Status: Done] **Dynamic Cost Reduction**: `ModifierDef` および `CostModifier` に `value_reference` フィールドを追加し、`ManaSystem` にて動的な値（`CARDS_DRAWN_THIS_TURN` 等）に基づくコスト計算ロジックを実装しました。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Directory Restructuring**: `python/gui` を `dm_toolkit/gui` へ移動しました。
*   [Status: Done] **Encoding Audit**: ソースコードのShift-JIS対応完了。
*   [Test: Pass] **GUI Tests (Headless)**: CI環境向けのスキップロジック実装済み。
*   [Status: Done] **ModifierEditForm Update**: スタティックアビリティ編集画面（`ModifierEditForm`）に「値参照（Value Reference）」の選択プルダウンを追加し、固定値以外の動的参照設定を可能にしました。

### 2.3 テスト環境 (`tests/`)
*   [Status: Done] **Directory Consolidation**: `python/tests/` を `tests/` へ統合。
*   [Test: Fail] **Dynamic Cost Reduction Verification**: `tests/verify_dynamic_cost_reduction.py` は検証中ですが、Pythonバインディング経由での `ManaSystem` 呼び出しに一部型不整合の課題が残っています（実装自体はC++側で完了）。

### 2.4 AI & 学習基盤 (`src/ai` & `dm_toolkit/training`)
*   [Status: WIP] **Smart Evolution Scoring**: `evolution_ecosystem.py` に「使用頻度」と「リソース使用」に基づく評価ロジックを追加しました。
    *   **実装**: `collect_smart_stats` 関数により、`GameInstance` を用いた統計収集ループを実装。

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] Trigger Stack Stabilization (トリガースタックの安定化)
[Status: Review]
パイプライン制御の安定化は完了。引き続きエッジケースの検証を行います。

### 3.2 [Priority: High] Dynamic Cost & Stat Integration (動的コストと統計統合)
[Status: Review]
動的コスト軽減のコアロジックは実装完了しました。

1.  [Status: Todo] **Verification Script Fix**: `ManaSystem` のPythonバインディングにおける型変換の問題を解決し、検証スクリプトをパスさせる。
2.  [Status: Todo] **Self-Targeting Logic**: 現在の `TargetUtils` ロジックと `CostModifier` の相互作用（特に `owner="SELF"` フィルタの挙動）を詳細に検証する。

### 3.3 [Priority: High] Evolution Ecosystem Refinement (進化エコシステムの改善)
[Status: WIP]
スマートスコアの実装を行いました。次は統計収集の不具合修正とパフォーマンス改善です。

1.  [Status: Todo] **Stats Aggregation Debug**: Python側で統計値が正しく集計されない問題（ID不整合等）を修正する。
2.  [Status: Todo] **C++ Stats Migration**: 現在Pythonで行っている低速な統計収集ループを、C++の `ParallelRunner` に統合し、高速化・安定化を図る。

## 4. 今後の課題 (Future Tasks)

1.  [Status: Todo] **Phase 7 Implementation**: 新JSONスキーマへの移行。
2.  [Status: WIP] **Binding Restoration**: 残るテストケースの修正。

## 5. 運用ルール (Operational Rules)
*   **テストコードの配置**: すべてのテストコード（Python）はプロジェクトルートの `tests/` ディレクトリに集約する。`python/tests/` などの他のディレクトリには新規テストを作成しないこと。
*   **CI遵守**: `PyQt6` 依存テストはスキップし、必ずCIがグリーンになる状態でマージする。
*   **バインディング追従**: C++変更時は必ず `src/bindings/bindings.cpp` を更新する。
