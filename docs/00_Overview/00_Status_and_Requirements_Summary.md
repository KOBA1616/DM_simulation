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
*   **EffectResolver (Legacy)**: 現在の主力ロジック。Phase 6で廃止予定。
*   **GameCommand (Architecture)**: 基盤実装および `GameState` への統合が完了。Pythonバインディングを通じて `execute/undo` が動作することを確認済み。
*   **汎用コストシステム**: 実装済み。新エンジンでもそのまま利用する。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Status**: 稼働中 (Ver 2.3)。
*   **Freeze**: エンジン刷新に伴うデータ構造の変更が確定するまで、機能追加および改修を凍結する。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   **Status**: パイプライン構築済み。
*   **Pending**: エンジン刷新による破壊的変更を避けるため、新エンジン稼働まで学習プロセス（Phase 3.2）は待機とする。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

現在の最優先タスクは「エンジンの刷新」です。これが完了するまで他のタスクはブロックされます。

### 3.1 [Priority: Critical] Phase 6: エンジン刷新 (Engine Overhaul)
**Status: In Progress**
`EffectResolver` を解体し、イベント駆動型システムと命令パイプラインへ完全移行します。

*   **Step 1: イベント駆動基盤の実装**
    *   `TriggerManager`: シングルトン/コンポーネントによるイベント監視・発行システムの実装。（実装完了）
    *   `check_triggers` メソッドによる「Passive -> Triggered -> Interceptor」の順次チェック機構を実装済み。
*   **Step 2: 命令パイプライン (Instruction Pipeline) の実装**
    *   `PipelineExecutor` (VM) の基盤実装完了。
    *   `LegacyJsonAdapter` により従来のJSONデータを `Instruction` 列に変換可能。
*   **Step 3: GameCommand への統合**
    *   **Completed**: `TransitionCommand` (移動), `MutateCommand` (状態変更), `FlowCommand` (制御) 等のコアコマンドを実装し、`GameState` に統合済み。Undo機能の動作を確認。
    *   **In Progress**: `EffectResolver` の各ロジックを `GameCommand` 発行に置き換えていくリファクタリング。
*   **Step 4: 移行と検証**
    *   既存テストケースの新エンジン上でのパス確認。

### 3.2 [Pending] Phase 3.2: AI 本番運用 (Production Run)
**Status: On Hold (Waiting for Phase 6)**
エンジン刷新完了後、新アーキテクチャ上でAI学習を再開します。

### 3.3 [Frozen] Phase 5: エディタ機能の完成 (Editor Polish)
**Status: Frozen**
エンジン刷新完了後、必要に応じてデータ構造の変更をエディタに反映させます。

---

## 4. 汎用コストおよび支払いシステム (General Cost and Payment System)

（変更なし。Phase 6においても `CostPaymentSystem` は継続利用する。）

---

## 5. イベント駆動型トリガーシステム詳細要件 (Event-Driven Trigger System Specs)

（変更なし）

---

## 6. イベント駆動型アクション・リアクション詳細要件 (Event-Driven Action/Reaction Specs)

（変更なし）

---

## 7. GameCommand アーキテクチャ詳細設計 (GameCommand Architecture Specs)

**Status: Implemented (Core)**

全ての状態変更操作を「コマンド」としてカプセル化し、Undo/Redo とログ記録を統一しました。

### 7.1 基本命令セット (Primitives)

1.  **CMD_TRANSITION (TransitionCommand)**:
    *   カードのゾーン移動。
    *   実装済み。`GameState` の `card_owner_map` 更新も一元管理。
2.  **CMD_MUTATE (MutateCommand)**:
    *   タップ/アンタップ、パワー変更、シールドブレイク、効果付与。
    *   実装済み。数値または文字列によるパラメータ指定に対応。
3.  **CMD_FLOW (FlowCommand)**:
    *   フェーズ遷移、ターン終了、ステップ移行。
    *   実装済み。
4.  **CMD_QUERY (QueryCommand)**:
    *   エンジンからエージェントへの入力要求。
    *   実装済み。
5.  **CMD_DECIDE (DecideCommand)**:
    *   エージェントからの回答。
    *   実装済み。

### 7.2 移行要件

*   `GameState` に `execute_command()` と `undo_last_command()` を実装済み。
*   既存の `Handler` クラス (`MoveCardHandler`, `DestroyHandler` 等) を `GameCommand` を使用するように更新済み。

---

## 8. 命令パイプライン (Instruction Pipeline) 詳細要件

（変更なし）

---

## 9. 移行戦略 (Migration Strategy)

1.  **基盤実装**: 完了 (`TriggerManager`, `PipelineExecutor`, `GameCommand`).
2.  **ラッパー作成**: 完了 (`LegacyJsonAdapter`).
3.  **段階的置換**: `EffectResolver` ロジックの置き換えを継続する。
4.  **EffectResolverの廃止**: 最終フェーズ。

---

## 10. 将来的な理想アーキテクチャ案 (Future Scope)

（変更なし）
