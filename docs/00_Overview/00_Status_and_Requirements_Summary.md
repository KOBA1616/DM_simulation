# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。
現在、Phase 0（基盤構築）、Phase 1（エディタ・エンジン拡張）、Phase 2（不完全情報対応）、および **Phase 4（アーキテクチャ刷新）の実装** を完了しました。

今後は **Phase 6（GameCommandアーキテクチャ・エンジン刷新）** を最優先事項とし、イベント駆動型システムへの移行とエンジンの汎用化を進めます。エディタの機能改善（バリデーション等）は、エンジン刷新後の構造に合わせて実施します。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   **フルスペック実装**: 基本ルールに加え、革命チェンジ、侵略、ハイパーエナジー、ジャストダイバー、ツインパクト、封印（基礎）、呪文ロックなどの高度なメカニクスをサポート済み。
*   **整合性と安定性の向上**: データ構造の統一、終了処理の安定化、クリーンアップAPIの導入完了。
*   **汎用コストシステム（統合完了）**: `CostPaymentSystem` を実装し、エンジンに統合済み。
*   **アクションシステム**: `IActionHandler` による完全なモジュラー構造。
*   **高速シミュレーション**: OpenMPによる並列化により、秒間数千〜数万試合の自己対戦が可能。
*   **イベント駆動・コマンド基盤**: `GameCommand` および `TriggerManager` の導入完了。

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
*   **DataCollector**: C++バインディングを介して `collect_data_batch` および `ParallelRunner` をPythonから利用可能。

### 2.4 サポート済みアクション・トリガー一覧 (Supported Actions & Triggers)
（変更なし：`EffectActionType` および `TriggerType` は現行コードベースに準拠）

### 2.5 実装上の不整合・未完了項目 (Identified Implementation Inconsistencies)
*   現在、主要な不整合は解消されました。

### 2.6 現在の懸念事項と既知の不具合 (Current Concerns and Known Issues)
*   `PlayHandler` において `TransitionCommand` を用いたStack移動処理への移行が完了しましたが、AI学習時のアクション空間エンコーディング (`ActionEncoder`) との整合性確認が必要です。現時点では `ActionEncoder` は `core::Action` に依存しています。

※ 完了した詳細な実装タスクは `docs/00_Overview/99_Completed_Tasks_Archive.md` にアーカイブされています。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

エンジンの根本的な刷新（GameCommand化）を優先し、その後にエディタやAI本番運用を進める方針でロードマップを再編しました。

### 3.0 [Priority: High] Phase 6: GameCommand アーキテクチャとエンジン刷新 (Engine Overhaul)

AI学習効率と拡張性を最大化するため、エンジンのコアロジックを「イベント駆動型」かつ「5つの基本命令 (GameCommand)」に基づくアーキテクチャへ刷新します。
メテオバーン、超次元、その他未実装の特殊メカニクスは、このフェーズにおける「アクションの汎用化」によって自然に解決・実装されます。

1.  **イベント駆動型トリガーシステムの実装**
    *   ハードコードされたフックポイントを廃止し、`TriggerManager` による一元管理へ移行。
    *   **Status**: `TriggerManager`, `GameEvent` クラスの実装とPythonバインディングが完了 (Phase 6.1 Completed)。
2.  **GameCommand (Primitives) の実装**
    *   全てのアクションを `TRANSITION`, `MUTATE`, `FLOW`, `QUERY`, `DECIDE` に分解・再実装。
    *   **Status**: 基本5命令のクラス実装、Pythonバインディング、および `GameState` への統合が完了。Unit Test (`tests/test_game_command.py`) を復元・実装し動作確認済み (Phase 6.2 Completed)。
3.  **アクション汎用化**
    *   **Status**: `MOVE_CARD`、`TAP`、`UNTAP`、`APPLY_MODIFIER`、`MODIFY_POWER`、`BREAK_SHIELD`、`DESTROY_CARD`、`PLAY_CARD`、および `ATTACK` (AttackHandler) のハンドラを `GameCommand` を使用するように移行完了。`PlayHandler` におけるStack移動処理も `TransitionCommand` へ移行済み (Phase 6.3 Completed)。
    *   **Next**: 完了したGameCommandアーキテクチャを用いたAI学習の再開（Phase 3.2へ移行）。

### 3.1 [Priority: Medium] Phase 5: エディタ機能の完成 (Editor Polish & Validation)

エンジン刷新後、新しいデータ構造に合わせてエディタのバリデーションを強化します。

1.  **Logic Mask (バリデーション) の実装**
    *   公式ルールに基づく最小限のマスク処理を実装。過度な制限は設けず、明らかな矛盾のみを防ぐ。
    *   **ルール**:
        *   **呪文 (Spell)**: 「パワー」フィールドを無効化（0固定）。
        *   **進化クリーチャー**: 「進化条件」の設定を有効化。
        *   **その他**: 基本的に制限なし（ユーザーの自由度を確保）。
    *   **Status**: 実装完了。`CardEditForm` にてタイプ別のUI表示切替とデータ保存ロジック（呪文のパワー0固定、進化条件の保存）を実装 (Phase 5.1 Completed)。

### 3.2 [Priority: Post-Phase 6] AI 本番運用 (Production Run)

「最強のAI」を目指す大規模学習は、GameCommandアーキテクチャの完成後に行います。

*   **タイミング**: Phase 6 完了後（現在、着手可能な状態）。
*   **理由**: GameCommandによるアクション空間の圧縮（意味ベースの学習）が完了した状態で学習させることで、未知のカードへの汎化性能と学習効率が劇的に向上するため。
*   **現在のステータス**: `ParallelRunner` および `DataCollector` のC++実装がPythonバインディング経由で利用可能であることを確認済み。`collect_training_data.py` は現在Python側でのループ実装となっているため、C++側の高速収集ロジックへの切り替えが次の具体的なタスクとなる。
*   **Next**: `collect_training_data.py` を更新し、C++側の高速データ収集パイプラインを利用するように変更する。また、`ActionEncoder` のGameCommand対応状況を精査する。

---

## 4. 汎用コストおよび支払いシステム (General Cost and Payment System)

（変更なし）

---

## 5. イベント駆動型トリガーシステム詳細要件 (Event-Driven Trigger System Specs)

Existing contents unchanged.

---

## 6. イベント駆動型アクション・リアクション詳細要件 (Event-Driven Action/Reaction Specs)

Existing contents unchanged.

---

## 7. GameCommand アーキテクチャ詳細設計 (GameCommand Architecture Specs)

Existing contents unchanged.

---

## 8. 命令パイプラインと汎用アクション構造 (Instruction Pipeline & Generalized Action Structure)

Existing contents unchanged.

---

## 9. 移行と互換性戦略 (Migration & Comparison Strategy)

Existing contents unchanged.

---

## 10. 将来的な理想アーキテクチャ案 (Ideal Architecture Proposal)

Existing contents unchanged.
