# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## ステータス定義 (Status Definitions)
*   `[Status: Todo]` : 未着手。
*   `[Status: WIP]` : 作業中。
*   `[Status: Review]` : 実装完了、レビュー待ち。
*   `[Status: Done]` : 完了・マージ済み。
*   `[Status: Blocked]` : 停止中。
*   `[Status: Deferred]` : 延期。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

現在、**Phase 1 (Engine Base)**, **Phase 6 (Command Architecture)**, **Phase 7 (Hybrid Engine)** の実装が完了し、エンジンの刷新が終了しました。
これに伴い、**Phase 2 (Inference)** および **Phase 3 (Evolution)** のAI開発フェーズを再開しています。
C++コア（`dm_ai_module`）は `GameCommand` ベースの堅牢な設計となり、並列対戦環境 (`ParallelRunner`) も稼働しています。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **GameCommand Architecture (Phase 6)**: `Action` を `GameCommand` (Transition, Mutate, Flow, etc.) に分解し、イベント駆動型アーキテクチャへ移行完了。
*   [Status: Done] **GenericCardSystem**: `EffectResolver` を代替し、`IActionHandler` によるモジュラーな処理系を確立。
*   [Status: Done] **Card Owner Refactor**: `CardInstance` に `owner` フィールドを追加し、コントローラー判定をO(1)化。
*   [Status: Done] **Multi-Civilization**: 多色カード（`std::vector<Civilization>`）のサポートおよび厳密なマナ支払いロジックの実装完了。
*   [Status: Done] **Advanced Mechanics**: 革命チェンジ、ハイパー化 (Hyper Energy)、ジャストダイバー、Condition Systemの実装完了。

### 2.2 AI & ソルバー (`src/ai`)
*   [Status: Done] **Parallel Runner**: OpenMPを用いた並列対戦環境の実装完了。
*   [Status: Done] **Lethal Solver**: ヒューリスティックベースの詰み判定 (`LethalSolver`) 実装完了。
*   [Status: Done] **MCTS & Evaluator**: AlphaZero準拠のMCTSおよび `BeamSearchEvaluator` の実装完了。
*   [Status: Done] **Inference Core (Phase 2)**: 相手手札推論を行う `PimcGenerator`, `DeckInference` のC++実装完了。
*   [Status: Todo] **Transformer Model (Phase 4)**: C++側 (`TensorConverter`) の準備は完了しているが、Python側のモデル実装および学習スクリプトは未実装。

### 2.3 カードエディタ & ツール (`python/gui`)
*   [Status: Done] **Card Editor V2**: JSONツリー編集、変数リンク (Variable Linking)、Condition編集機能の実装完了。
*   [Status: Done] **Visualization**: シミュレーション実行ダイアログ、日本語ローカライズ完了。
*   [Status: WIP] **Logic Mask**: ルール矛盾（呪文のパワー設定など）を防ぐバリデーション機能の実装中。

### 2.4 学習基盤 (`python/training`)
*   [Status: Done] **Training Pipeline**: `collect_training_data.py` -> `train_simple.py` の基本ループ稼働中。
*   [Status: WIP] **Evolution Ecosystem (Phase 3)**: デッキ自動進化 (`verify_deck_evolution.py` 等) のパイプライン統合が進行中。

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: High] AI Evolution (AI進化) - Phase 2 & 3
エンジン刷新が完了したため、AIの「賢さ」と「メタゲーム適応」に焦点を戻します。

1.  **Deck Evolution (Phase 3)**:
    *   `verify_deck_evolution.py` をベースに、PBT (Population Based Training) によるデッキ最適化ループを完全自動化する。
    *   `meta_decks.json` を動的に更新するシステムを構築する。

2.  **Inference Integration (Phase 2)**:
    *   C++で実装された `DeckInference` をPythonの学習ループおよび対戦エージェントに組み込み、不完全情報ゲームとしての強さを向上させる。

### 3.2 [Priority: Medium] Advanced AI Architecture - Phase 4
Transformerモデルへの移行を進めます。

1.  **Python Implementation**: `TensorConverter` (C++) が出力するシーケンスデータに対応したPyTorchモデル (`TransformerEvaluator`) を実装する。
2.  **Training Update**: 学習スクリプトをTransformer対応に更新する。

### 3.3 [Priority: Low] Engine Refinement
1.  **Strict Lethal Solver**: ヒューリスティックではない、完全探索によるLethal Solverへの昇華。
2.  **Reaction Ability Editor**: ニンジャ・ストライク等のリアクション編集UIの実装。

## 4. 運用ルール (Operational Rules)
*   **コミットメッセージ**: 日本語で記述する。
*   **コード規約**: PythonはPEP8準拠、C++はGoogle Style準拠。
*   **テスト**: 機能追加時は必ず `python/tests/` に単体テストを追加し、`pytest` をパスさせること。
