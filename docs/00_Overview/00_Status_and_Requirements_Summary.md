# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

Phase 6のエンジン刷新が完了し、プロジェクトのフォーカスは「AIの進化と強化」へと移行しました。
現在は「Duel Masters AI Comprehensive Implementation Plan」に基づき、確実性の高いソルバー実装から自己進化するエコシステムの構築を目指します。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   **Status: Modernized & Stable**
*   **GameCommand**: 全てのアクションが `GameCommand` (Transition, Mutate, Flow, Query, Decide) に統一され、Undo/Redo基盤が確立済み。
*   **Event-Driven Trigger**: `TriggerManager` によるイベント駆動型トリガーシステムが稼働中。
*   **Action Generalization**: 各種アクション（移動、破壊、ドロー等）は汎用ハンドラ (`IActionHandler`) に委譲されている。
*   **PipelineExecutor**: JSON定義された命令列を実行するパイプラインシステムが実装済み。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Status: Active (Ver 2.3)**
*   **Features**:
    *   `LogicTreeWidget` による階層的な効果編集。
    *   `ScenarioEditor` による対戦シナリオの管理。
    *   `ActionEditForm`, `CardEditForm` による詳細なパラメータ設定。
    *   日本語ローカライゼーション対応済み。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`, `src/ai`)
*   **Status: Phase 1-2 In Progress**
*   **Solver**: `LethalSolver` (Heuristic) 実装済み。
*   **Evaluation**: `BeamSearchEvaluator` 実装済み。
*   **Evolution**: `DeckEvolution` (Self-Evolution) スタブ実装済み。
*   **Training**: `collect_training_data.py`, `train_simple.py` パイプライン稼働中。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

プロジェクトは現在、AIの質的向上と自己進化システムの構築を主眼とした「AI Comprehensive Implementation Plan」に従って進行しています。

### 3.1 [Priority: High] Phase 1: 確定完全情報の掌握 (Certainty / Lethal Solver)
**Status: Mostly Completed (Verification Ongoing)**
確率に頼らない、詰み盤面（Lethal）を正確に解くソルバーの実装。
*   **Goal**: 複雑な盤面（ブロッカー、S・トリガーの可能性、革命チェンジ等）において、確実な勝利ルートを発見する。
*   **Tech**:
    *   `LethalSolver`: 貪欲法および探索による詰み判定（実装済み）。
    *   **Next**: 探索深さの拡張と、より高度な妨害（S・トリガー）を考慮した分岐処理。

### 3.2 [Priority: Medium] Phase 2: 不完全情報の推定 (Imperfect Information / PIMC)
**Status: Planned**
「神の視点（God View）」と「プレイヤー視点」のギャップを埋めるための推論システム。
*   **Goal**: 相手の手札やシールドの内容を確率的に推定し、リスク管理を行う。
*   **Tech**:
    *   **PIMC (Perfect Information Monte Carlo)**: 相手の隠された情報をサンプリングして複数の「確定情報ゲーム」として解き、その平均を取る手法。
    *   **Opponent Hand Inference Model**: 相手のマナ埋めやプレイ履歴から手札を予測するモデル。

### 3.3 [Priority: Medium] Phase 3: 自己進化エコシステム (Self-Evolution / AlphaZero + PBT)
**Status: In Progress**
人間の介入なしにAIが自らデッキとプレイングを進化させるシステムの構築。
*   **Goal**: 環境（メタゲーム）の変化に合わせて、AIが自動的にデッキを調整し、強くなる。
*   **Tech**:
    *   **DeckEvolution**: C++による高速なデッキ変異・交差ロジック（スタブ実装済み）。
    *   **Population Based Training (PBT)**: 複数のエージェント（デッキ＋重み）を戦わせ、勝者を残して変異させる学習ループ。
    *   **ParallelRunner**: 大規模並列対戦基盤（実装済み）。

### 3.4 [Priority: Low] Phase 4: アーキテクチャの刷新 (Architecture / Linear Attention)
**Status: Research / Partially Implemented**
長期的なゲーム展開（履歴）を効率的に扱うためのモデル構造の刷新。
*   **Goal**: 従来のResNet/CNNベースから、可変長のゲーム履歴を扱えるTransformerベースへ移行。
*   **Tech**:
    *   **Linear Attention**: 計算コストを抑えつつ長期間の依存関係を学習する。
    *   **NetworkV2**: 実装済みだが、学習パイプラインへの完全統合は未完了。

### 3.5 [Priority: Low] Phase 5: 未来のAI (Future Refinement / PPO + LSTM)
**Status: Future**
AlphaZero（MCTS）の限界を超えるための強化学習手法の導入。
*   **Goal**: 探索コストを削減し、直感（Policy）だけで強い動きができるAI。
*   **Tech**:
    *   **PPO (Proximal Policy Optimization)**: 方策勾配法による安定した学習。
    *   **LSTM / State-Space Models**: 時系列データのより高度な処理。

---

## 4. 最近の達成事項 (Recent Achievements) - Archive of Phase 6

以下のタスクは完了し、現在のAI開発の強固な基盤となっています。

*   **Engine Overhaul**:
    *   `EffectResolver` の複雑なswitch文ロジックを解体し、`GameCommand` と `PipelineExecutor` へ移行。
    *   イベント駆動型 `TriggerManager` の導入により、S・トリガーや常在効果の処理を適正化。
*   **Action Generalization**:
    *   カード効果を「原子的なアクション（移動、修正、選択）」に分解し、汎用ハンドラで処理することで、新規カード実装コストを大幅に削減。
*   **Scenario System**:
    *   `ScenarioEditor` と `ScenarioExecutor` により、特定の盤面（詰将棋的な状況）からの学習と評価が可能になった。
*   **Reaction System**:
    *   ニンジャ・ストライクや革命チェンジのような「相手ターン中の行動」を正規化されたルール処理として実装完了。

