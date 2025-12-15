# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。
現在、Phase 0（基盤構築）、Phase 1（エディタ・エンジン拡張）、Phase 2（不完全情報対応）、および **Phase 4（アーキテクチャ刷新）の実装** を完了しました。

Phase 6（GameCommandアーキテクチャ・エンジン刷新）の実装が完了し、エンジンの安定性向上とイベント駆動型システムへの移行が達成されました。
今後は **Phase 3.2（AI本番運用）** を最優先事項とし、大規模な自己対戦によるAI強化サイクルを開始します。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   **フルスペック実装**: 基本ルールに加え、革命チェンジ、侵略、ハイパーエナジー、ジャストダイバー、ツインパクト、封印（基礎）、呪文ロックなどの高度なメカニクスをサポート済み。
*   **イベント駆動型アーキテクチャ**: `TriggerManager` と `PendingEffect` を中心としたスタックベースの解決システムを実装。
    *   **ループ防止 (Loop Prevention)**: `chain_depth` による無限ループ検出を実装済み。
    *   **リアクションウィンドウ**: 非同期リアクション待機状態 (`waiting_for_reaction`) をサポート。
    *   **コンテキスト参照**: 動的な値参照 (`context_val_key`) をサポート。
*   **汎用コストシステム（統合完了）**: `CostPaymentSystem` を実装し、エンジンに統合済み。
*   **アクションシステム**: `GameCommand` (Primitives) に完全移行し、アクションの汎用化とUndo対応を強化。
*   **高速シミュレーション**: OpenMPによる並列化により、秒間数千〜数万試合の自己対戦が可能。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Card Editor Ver 2.3**: 3ペイン構成（ツリー/プロパティ/プレビュー）。
    *   **テキスト生成**: 数値範囲や任意選択の日本語生成ロジック強化済み。
    *   **リアクション編集**: `ReactionWidget` による動的UI切り替えをサポート。
    *   **ロジックマスク (Phase 5)**: カードタイプに応じた入力制限（呪文のパワー無効化、進化条件の有効化）を実装済み。
*   **機能**: JSONデータの視覚的編集、ロジックツリー、変数リンク、テキスト自動生成、デッキビルダー、シナリオエディタ。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   **AlphaZero Pipeline**: データ収集 -> 学習 -> 評価 の完全自動ループが稼働中。
*   **推論エンジン**: 相手デッキタイプ推定 (`DeckClassifier`) と手札確率推定 (`HandEstimator`) を実装済み。
*   **探索アルゴリズム**: MCTSおよびBeam Search（決定論的探索）を実装済み。
*   **ONNX Runtime (C++) 統合**: `NeuralEvaluator` によるC++内での高速推論をサポート。
*   **Phase 4 アーキテクチャ (実装完了)**:
    *   **NetworkV2**: Transformer (Linear Attention) ベースの可変長入力モデルを実装。
    *   **TensorConverter**: C++側でのシーケンス変換ロジックを実装済み。

### 2.4 サポート済みアクション・トリガー一覧 (Supported Actions & Triggers)
（変更なし：`EffectActionType` および `TriggerType` は現行コードベースに準拠）

### 2.5 実装上の不整合・未完了項目 (Identified Implementation Inconsistencies)
*   特になし。主要なアーキテクチャ移行タスクは完了しました。

### 2.6 現在の懸念事項と既知の不具合 (Current Concerns and Known Issues)
*   AIの学習リソース（計算時間）の確保と、大規模学習時の安定性検証が今後の課題です。

※ 完了した詳細な実装タスクは `docs/00_Overview/99_Completed_Tasks_Archive.md` にアーカイブされています。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

エンジンの根本的な刷新（GameCommand化）が完了しました。

### 3.0 [Priority: High] Phase 6: GameCommand アーキテクチャとエンジン刷新 (Engine Overhaul)

AI学習効率と拡張性を最大化するため、エンジンのコアロジックを「イベント駆動型」かつ「5つの基本命令 (GameCommand)」に基づくアーキテクチャへ刷新します。

1.  **イベント駆動型トリガーシステムの実装**
    *   ハードコードされたフックポイントを廃止し、`TriggerManager` による一元管理へ移行。
    *   **Status**: `TriggerManager`, `GameEvent` クラスの実装とPythonバインディングが完了 (Phase 6.1 Completed)。
    *   **詳細要件**: ループ防止 (`chain_depth`)、コンテキスト参照 (`context_val_key`) の実装完了 (Phase 6.4 Completed)。
2.  **GameCommand (Primitives) の実装**
    *   全てのアクションを `TRANSITION`, `MUTATE`, `FLOW`, `QUERY`, `DECIDE` に分解・再実装。
    *   **Status**: 基本5命令のクラス実装、Pythonバインディング、および `GameState` への統合が完了。Unit Test (`tests/test_game_command.py`) を復元・実装し動作確認済み (Phase 6.2 Completed)。
3.  **アクション汎用化**
    *   **Status**: 全ハンドラの `GameCommand` 移行完了。リアクションウィンドウ (`waiting_for_reaction`) の基盤実装完了 (Phase 6.3 & 6.5 Completed)。
    *   **Completed**: GameCommandアーキテクチャへの移行は完了しました。

### 3.1 [Priority: High] Phase 3.2: AI 本番運用 (Production Run)

GameCommandアーキテクチャによるエンジン刷新が完了したため、AI学習パイプラインの検証と本番運用を開始します。

*   **現在の状況**:
    *   学習パイプライン (`collect_training_data.py`, `train_simple.py`) の動作確認完了。
    *   評価スクリプト (`verify_performance.py`) の `ActionEncoder` サイズ不整合を修正し、正常動作を確認済み。
    *   **Next**: 大規模な自己対戦（数百万ゲーム規模）とモデルの継続的なアップデートを実施します。

### 3.2 [Priority: Medium] Phase 5: エディタ機能の完成 (Editor Polish & Validation)

エンジン刷新後、新しいデータ構造に合わせてエディタのバリデーションを強化します。

1.  **Logic Mask (バリデーション) の実装**
    *   公式ルールに基づく最小限のマスク処理を実装。過度な制限は設けず、明らかな矛盾のみを防ぐ。
    *   **Status**: 実装完了 (Phase 5.1 Completed)。

---

## 4. 汎用コストおよび支払いシステム (General Cost and Payment System)

（変更なし）

---

## 5. イベント駆動型トリガーシステム詳細要件 (Event-Driven Trigger System Specs)

既存の `EffectResolver` を刷新するための技術要件。以下の項目は実装完了しました。

### 5.1 基本アーキテクチャ (Architecture)
*   **TriggerManager**: 実装完了。
*   **Event Object**: 実装完了。

### 5.2 詳細要件 (Detailed Requirements)

1.  **Event Monitor (B案: イベント監視型)**
    *   **Status**: 実装完了。

2.  **Loop Prevention (ループ防止)**
    *   **仕様**: `PendingEffect` に `chain_depth` カウンタを持たせ、閾値（50）を超えたら解決をスキップする。
    *   **Status**: 実装完了 (`game_state.hpp` / `generic_card_system.cpp`)。

3.  **Context Reference (コンテキスト参照)**
    *   **仕様**: `FilterDef` に `context_val_key` を追加し、動的参照を可能にする。
    *   **Status**: 実装完了 (`card_json_types.hpp`)。

---

## 6. イベント駆動型アクション・リアクション詳細要件 (Event-Driven Action/Reaction Specs)

### 6.1 基本フロー (Basic Flow)
イベント発生 -> トリガー検知 -> 保留効果(PendingEffect)生成 -> 優先権に基づく解決。

### 6.2 詳細要件 (Detailed Requirements)

1.  **Reaction Window (A案: 非同期・ステートマシン型)**
    *   **仕様**: リアクション待機時、`waiting_for_reaction` フラグを立てて制御を返す。
    *   **Status**: 実装完了 (`GameState` にフラグ追加、`ReactionSystem` 改修)。

2.  **Interceptor Layer (置換効果レイヤー)**
    *   **仕様**: アクション実行直前に介入する **Interceptor** として実装する。
    *   **Status**: Phase 6.3 でのアクションハンドラ汎用化により基盤は整っているが、具体的な置換効果（破壊置換など）の実装はカードごとの個別実装として扱う。

3.  **Optional vs Mandatory (任意と強制)**
    *   **仕様**: JSON定義に `optional: true/false` を持たせる。
    *   **Status**: `ActionDef.optional` として実装完了。

---

（以降、アーキテクチャ設計等は変更なし）
