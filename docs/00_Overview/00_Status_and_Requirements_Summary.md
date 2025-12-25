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
AI領域では、並列対戦環境の構築が完了し、次は不完全情報ゲームへの対応（Phase 2）とデッキ自動進化システム（Phase 3）の統合フェーズに移行します。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **Card Owner Refactor**: `CardInstance` 構造体に `owner` フィールドを追加し、コントローラー判定をO(1)化。
*   [Status: Done] **EffectResolver Removal**: モノリシックな `EffectResolver` を削除し、処理を `IActionHandler` と `GameLogicSystem` へ委譲完了。
*   [Status: Done] **Action Generalization**: `GenericCardSystem` によるアクション処理の統一化完了。
*   [Status: Done] **Revolution Change**: 革命チェンジ（`ON_ATTACK_FROM_HAND` トリガーおよび入替処理）の実装完了。
*   [Status: Done] **Hyper Energy**: ハイパー化（コスト軽減およびクリーチャータップ）の実装完了。
*   [Status: Done] **Just Diver**: 「ジャストダイバー」などのターン経過依存の耐性ロジック実装完了。
*   [Status: Done] **Condition System**: `ConditionDef` および `ConditionSystem` による汎用的な発動条件判定の実装完了。

### 2.2 AI & ソルバー (`src/ai`)
*   [Status: Done] **Lethal Solver (DFS)**: 基本的な詰み判定（`LethalSolver`）の実装完了。ただし現在はヒューリスティックベースであり、厳密解ではない。
*   [Status: Done] **Parallel Runner**: OpenMPを用いた並列対戦環境 (`ParallelRunner`) の実装完了。`verify_performance.py` によるバッチ推論との連携を確認済み。
*   [Status: Done] **MCTS Implementation**: AlphaZero準拠のモンテカルロ木探索実装完了。
*   [Status: WIP] **Inference System**: 相手の手札やシールド推論を行うシステムの設計段階（Phase 2）。

### 2.3 カードエディタ & ツール (`dm_toolkit/gui` / `python/gui`)
*   [Status: Done] **Card Editor V2**: データ構造をJSONツリーベースに刷新。
*   [Status: Done] **Variable Linking**: アクション間で変数を共有する「変数リンク機能」の実装完了。
*   [Status: Done] **Condition Editor**: GUI上で発動条件（Condition）を編集するフォームの実装完了。
*   [Status: Done] **Simulation GUI**: GUIから並列シミュレーションを実行するダイアログの実装完了。

### 2.4 学習基盤 (`python/training`)
*   [Status: Done] **Training Loop**: `collect_training_data.py` -> `train_simple.py` の基本ループ構築完了。
*   [Status: Done] **Scenario Runner**: `ScenarioExecutor` を用いた定石シミュレーション機能の実装完了。

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] AI Evolution System (AI進化システム)
[Status: WIP]
Phase 3の中核となる「自筆進化エコシステム」を構築します。

1.  [Status: Todo] **Deck Evolution (PBT)**: `verify_deck_evolution.py` のプロトタイプを拡張し、Population Based Trainingによるデッキ自動更新システムを実装する。
2.  [Status: Todo] **Meta-Game Integration**: 学習したモデルとデッキ情報を `meta_decks.json` に自動反映し、次世代の学習にフィードバックするループを構築する。

### 3.2 [Priority: High] Model Architecture (モデルアーキテクチャ)
[Status: Todo]
Phase 4の要件である高性能モデルへの移行を行います。

1.  [Status: Todo] **Transformer Implementation**: 現在のResNet/MLPモデルから、Self-Attentionを用いたTransformerモデルへ移行し、盤面の文脈理解能力を向上させる。

### 3.3 [Priority: Medium] Engine Robustness (エンジン堅牢化)
[Status: Todo]
エッジケースへの対応と安定性向上を図ります。

1.  [Status: Todo] **Strict Lethal Solver**: 現在のヒューリスティック版から、ルールを完全に厳密にシミュレートするソルバーへ移行する。
2.  [Status: Todo] **Memory Leak Fix**: `ParallelRunner` を長時間/大量スレッドで実行した際に発生する `std::bad_alloc` (メモリリーク) の調査と修正。

### 3.4 [Priority: Low] GUI Expansion (GUI拡張)
[Status: Deferred]

1.  [Status: Todo] **Reaction Ability Editor**: ニンジャ・ストライク等の `ReactionAbility` 編集UIの実装。
2.  [Status: Todo] **Logic Mask**: 矛盾する効果の組み合わせを防止するバリデーション機能。

## 4. 開発スケジュール・今後の予定 (Development Schedule)

1.  **直近の目標 (Phase 2-3 Integration)**:
    *   不完全情報推論（Inference System）の実装開始。
    *   デッキ進化システム（PBT）のパイプライン化。
2.  **中期的目標 (Phase 4)**:
    *   Transformerモデルの実装と学習。
3.  **長期的目標 (Phase 5)**:
    *   RNN/LSTMを用いた時系列データの活用と、PPOなどの強化学習アルゴリズムの導入。

## 5. 既知の問題 (Known Issues)

*   [Known Issue] **ParallelRunner Memory Leak**: 大規模な並列シミュレーション時にメモリリークが発生する可能性がある。
*   [Known Issue] **Inference Heuristic**: 現在の推論ロジックは単純な確率に基づいており、ブラフやメタゲームを考慮していない。

## 6. 運用ルール (Operational Rules)
*   **コミットメッセージ**: 日本語で記述する。
*   **ソースコード**: コメントも含めUTF-8で統一する。
*   **テスト**: `python/tests/` 以下のpytestを実行し、CIを通過させること。
