# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

現在、**Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** を最優先事項として進行中です。
既存のハードコードされた効果処理 (`EffectResolver`) を廃止し、イベント駆動型アーキテクチャと命令パイプライン (`Instruction Pipeline`) へ刷新することで、柔軟性と拡張性を確保します。

AI学習 (Phase 3) およびエディタ開発 (Phase 5) は、このエンジン刷新が完了するまで一時凍結します。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   **EffectResolver (Legacy)**: 現在の主力ロジック。巨大なswitch文により効果処理を行っているが、複雑化により限界に達している。そのため、game command方式に移行中。
*   **GameCommand (Partial)**: 基本クラスと一部のコマンドは実装済みだが、エンジンの中核ロジックとしては未統合。
*   **汎用コストシステム**: 実装済み。新エンジンでもそのまま利用する。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Status**: 稼働中 (Ver 2.3)。
*   **Freeze**: エンジン刷新に伴うデータ構造の変更が確定するまで、機能追加および改修を凍結する。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   **Status**: パイプライン構築済み。
*   **Pending**: エンジン刷新による破壊的変更を避けるため、新エンジン稼働まで学習プロセス（Phase 3.2）は待機とする。

---

### 3.1 [Priority: Critical] Phase 6: エンジン刷新 (Engine Overhaul)
**Status: In Progress**
`EffectResolver` を解体し、イベント駆動型システムと命令パイプラインへ完全移行します。

*   **Step 1: イベント駆動基盤の実装**
    *   **Status: Implemented**
    *   `TriggerManager`: シングルトン/コンポーネントによるイベント監視・発行システムの実装。（実装完了）
    *   `check_triggers` メソッドにより、`GameEvent` をトリガーとして `PendingEffect` を生成するフローを確立。
*   **Step 2: 命令パイプライン (Instruction Pipeline) の実装**
    *   **Status: Implemented & CIP Integrated**
    *   `PipelineExecutor` (VM) を実装済み。
    *   **New**: `GenericCardSystem` を更新し、`ON_PLAY` (CIP) トリガー処理を `PipelineExecutor` へルーティング開始。
    *   **New**: `LegacyJsonAdapter` を拡張し、`SEARCH_DECK` などの複雑なアクションを含む既存のJSONデータを命令列に変換可能にした。
    *   **Verify**: `tests/verify_cip_pipeline.py` にて標準的なCIP効果（ブロンズ・アーム・トライブ）および条件付き効果（マナ武装）がパイプライン経由で正しく動作することを確認済み。
*   **Step 3: GameCommand への統合**
    *   **Status: Completed**
    *   全てのアクションを `GameCommand` (Transition, Mutate, Flow等) 発行として統一し、Undo/Redo基盤を確立する。
    *   **New**: `GameInstance` にて `TriggerManager` を `GameState::event_dispatcher` と連携させ、コマンド実行時のイベント発行をトリガー検知につなげる統合を完了。
*   **Step 4: 移行と検証**
    *   **Known Issue**: `python/tests/test_mega_last_burst.py` が `DestroyHandler` の依存関係（CardRegistry vs CardDB）により失敗している。次フェーズで `GenericCardSystem::resolve_action` とハンドラの連携を詳細調査・修正する。
    *   **New**: S・トリガーおよび革命チェンジのイベント駆動フロー（リアクションウィンドウ生成）の検証完了。
