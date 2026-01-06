# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

## ステータス定義
*   `[Status: Todo]` : 未着手。
*   `[Status: WIP]` : 作業中。
*   `[Status: Review]` : 実装完了、レビュー待ち。
*   `[Status: Done]` : 完了・マージ済み。
*   `[Status: Blocked]` : 停止中。
*   `[Status: Deferred]` : 延期。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroおよびTransformerベースのAI学習環境を統合したプロジェクトです。

現在、**Core Engine (C++)** の実装はほぼ完了しており、以下のフェーズに焦点を移しています。
1.  **AI Evolution (Phase 2 & 3)**: PBTを用いたメタゲーム進化と推論システム。
2.  **Transformer Architecture (Phase 4)**: `dm_toolkit` によるシーケンスモデルの導入。
3.  **Editor Refinement**: カードエディタの完成度向上（Logic Mask等）。

## 2. 現行システムステータス (Current Status)

### 2.1 ゲームエンジン (`src/core`, `src/engine`)
*   [Status: Done] **Action/Command Architecture**: `GameCommand` ベースのイベント駆動モデル。
*   [Status: Done] **Advanced Mechanics**: 革命チェンジ (Revolution Change), ハイパー化 (Hyper Energy), ジャストダイバー等の実装完了。
*   [Status: Done] **Multi-Civilization**: 多色マナ支払いロジックの実装完了。
*   [Status: Done] **Stats/Logs**: `TurnStats` や `GameResult` の収集基盤。

### 2.2 AI システム (`src/ai`, `python/training`, `dm_toolkit`)
*   [Status: Done] **Parallel Runner**: OpenMP + C++ MCTS による高速並列対戦。
*   [Status: Done] **AlphaZero Logic**: MLPベースのAlphaZero学習ループ (`train_simple.py`).
*   [Status: WIP] **Transformer Model**: `DuelTransformer` (Linear Attention, Synergy Matrix) のクラス定義実装済み。学習パイプラインへの統合待ち。
*   [Status: WIP] **Meta-Game Evolution**: `evolution_ecosystem.py` によるデッキ自動更新ロジックの実装中。
*   [Status: Done] **Inference Core**: C++ `DeckInference` クラスおよびPythonバインディング実装済み。

### 2.3 開発ツール (`python/gui`)
*   [Status: Done] **Card Editor V2**: JSONツリー編集、変数リンク、Condition設定機能。
*   [Status: Done] **Simulation UI**: 対戦シミュレーション実行・可視化ダイアログ。
*   [Status: Todo] **Logic Mask**: カードデータ入力時の矛盾防止機能。

## 3. 次のステップ (Next Steps)

### 3.1 AI Implementation (Phase 3 & 4)
*   **Transformer Training Loop**: `dm_toolkit.ai.agent.transformer_model.DuelTransformer` を使用した学習スクリプト `train_transformer.py` の完成。
*   **Evolution Pipeline Integration**: `verify_deck_evolution.py` のロジックを本番の `evolution_ecosystem.py` に統合し、継続的な自己対戦環境を構築する。

### 3.2 Engine Maintenance
*   **Test Coverage**: 新機能（革命チェンジ、ハイパー化）に対するカバレッジの向上。
*   **Refactoring**: `src/engine` 内の古いロジックの清掃。

### 3.3 Documentation
*   **Update Specs**: 実装と乖離した古い要件定義書の更新（本タスクにて実施中）。

### 3.4 Command Pipeline / Legacy Action Removal
*   [Status: WIP] **旧Action完全削除ロードマップの遂行**: カードJSONの `actions` と関連する互換コード/UIを段階的に撤去し、`commands` を唯一の表現に統一する。
	*   ロードマップ: [docs/00_Overview/01_Legacy_Action_Removal_Roadmap.md](01_Legacy_Action_Removal_Roadmap.md)
	*   前提: `dm_toolkit.action_to_command.action_to_command` を唯一の Action→Command 入口にする（AGENTSポリシー準拠）。

## 4. ドキュメント構成
*   `docs/01_Game_Engine_Specs.md`: ゲームエンジンの詳細仕様。
*   `docs/02_AI_System_Specs.md`: AIモデル、学習パイプライン、推論システムの仕様。
*   `docs/03_Card_Editor_Specs.md`: カードエディタの機能要件。
*   `docs/00_Overview/archive/`: 過去の計画書や完了済みタスクのログ。
