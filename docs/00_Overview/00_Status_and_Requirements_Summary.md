# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。
現在、Phase 0（基盤構築）、Phase 1（エディタ・エンジン拡張）、Phase 2（不完全情報対応）、Phase 4（アーキテクチャ刷新）、および **Phase 6（GameCommandアーキテクチャ・エンジン刷新）** の実装を完了しました。

今後は **Phase 3.2（AI本番運用）** を最優先事項とし、GameCommandベースの新エンジンを用いたAIの学習サイクル確立を目指します。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   **フルスペック実装**: 基本ルールに加え、革命チェンジ、侵略、ハイパーエナジー、ジャストダイバー、ツインパクト、封印（基礎）、呪文ロックなどの高度なメカニクスをサポート済み。
*   **GameCommand アーキテクチャ (Phase 6 Completed)**:
    *   **イベント駆動**: `TriggerManager` によるイベント監視・発行システムの導入。
    *   **基本5命令**: `TRANSITION`, `MUTATE`, `FLOW`, `QUERY`, `DECIDE` による状態遷移の完全抽象化。
    *   **アクション汎用化**: 全アトミックアクションのGameCommandへの移行完了。
*   **汎用コストシステム**: `CostPaymentSystem` による複雑なコスト支払いロジックの統合。
*   **アクションシステム**: `IActionHandler` による完全なモジュラー構造。
*   **高速シミュレーション**: OpenMPによる並列化。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Card Editor Ver 2.3**: 3ペイン構成（ツリー/プロパティ/プレビュー）。
    *   **Logic Mask**: カードタイプに応じたバリデーション（呪文のパワー0固定等）を実装済み。
    *   **機能**: JSON編集、ロジックツリー、変数リンク、テキスト自動生成、デッキビルダー、シナリオエディタ。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   **AlphaZero Pipeline**: データ収集 -> 学習 -> 評価 のループ構造。
*   **推論エンジン**: 相手デッキタイプ推定 (`DeckClassifier`) と手札確率推定 (`HandEstimator`)。
*   **ONNX Runtime (C++) 統合**: `NeuralEvaluator` によるC++内推論。
*   **NetworkV2**: Transformer (Linear Attention) ベースのモデル。

### 2.4 サポート済みアクション・トリガー一覧 (Supported Actions & Triggers)
（変更なし：`EffectActionType` および `TriggerType` は現行コードベースに準拠）

### 2.5 実装上の不整合・未完了項目 (Identified Implementation Inconsistencies)
*   **GameCommand化に伴う学習パイプラインの不整合**: Phase 6完了後の統合テストにおいて、`collect_training_data.py` が実行時エラー（Segmentation Fault）を起こす問題が確認されています（後述）。

### 2.6 現在の懸念事項と既知の不具合 (Current Concerns and Known Issues)
*   **[Critical] `collect_training_data.py` Segmentation Fault**:
    *   **現象**: AI学習データ収集スクリプト実行時に `resolve_trigger: Instance 0 not found` エラーと共にクラッシュする。
    *   **原因**: エンジン内部で `Instance ID 0` （無効ID、あるいはデッキ内のカード）に対するトリガー解決が試みられており、`find_instance` が失敗している可能性がある。GameCommand化に伴うオブジェクトライフサイクル管理の変更が影響していると推測される。
    *   **ステータス**: 未解決。Phase 3.2（AI本番運用）のブロッキング要因となっている。

※ 完了した詳細な実装タスクは `docs/00_Overview/99_Completed_Tasks_Archive.md` にアーカイブされています。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

GameCommandアーキテクチャへの移行（Phase 6）が完了したため、AI学習環境の安定化と運用を最優先とします。

### 3.1 [Priority: Critical] Phase 3.2: AI 本番運用とデバッグ (Production Run & Debugging)

GameCommandベースのエンジンを用いたAI学習パイプラインを稼働させます。現在はクリティカルなバグによりブロックされています。

1.  **データ収集スクリプトの修正**
    *   `collect_training_data.py` のSegmentation Faultを修正する。
    *   `Instance 0` への不正アクセス箇所を特定し、トリガーシステムのID検証ロジックを強化する。
2.  **学習サイクルの検証**
    *   データ収集 -> 学習 (`train_simple.py`) -> 評価 (`verify_performance.py`) の完全なサイクルがエラーなく完了することを確認する。
3.  **大規模学習の実施**
    *   バグ修正後、数千エピソード規模のデータ収集と学習を実施し、ベースライン（Random/Heuristic）に対する勝率向上を確認する。

### 3.2 [Priority: Medium] Phase 5: エディタ機能の完成 (Editor Polish & Validation)

エンジン刷新後、新しいデータ構造に合わせてエディタのバリデーションを強化します。
（Logic Maskの実装はPhase 6と並行して完了済み）

---

## 4. 汎用コストおよび支払いシステム (General Cost and Payment System)

（変更なし）

---

## 5. イベント駆動型トリガーシステム詳細要件 (Event-Driven Trigger System Specs)

（実装完了済みのため、詳細はアーカイブまたはコードベース参照）

---

## 6. イベント駆動型アクション・リアクション詳細要件 (Event-Driven Action/Reaction Specs)

（実装完了済みのため、詳細はアーカイブまたはコードベース参照）

---

## 7. GameCommand アーキテクチャ詳細設計 (GameCommand Architecture Specs)

（実装完了済みのため、詳細はアーカイブまたはコードベース参照）

---

## 8. 命令パイプラインと汎用アクション構造 (Instruction Pipeline & Generalized Action Structure)

ハードコード（複合効果の個別C++実装）を撤廃し、あらゆるカード効果をデータ定義（JSON）のみで実現するための新アーキテクチャ。
※ Phase 6にて基盤は完成したが、完全移行は今後の課題。

### 8.1 概念 (Concept)
カードの効果を、単一のタイプではなく **「入出力を持つ命令 (Instruction) の連鎖」** として定義する。
各命令は **コンテキスト (Context)** と呼ばれる共有メモリを通じてデータの受け渡しを行う。

### 8.2 アーキテクチャ構成 (Architecture Components)

1.  **Instruction (命令)**
    *   最小単位の操作（SELECT, MOVE, CALCULATE, MODIFY）。
    *   **Args (引数)**: 定数、または変数参照（`$var_name`）を受け取る。
    *   **Out (出力)**: 実行結果を保存する変数名（`out: "$targets"`）。

2.  **Context (実行コンテキスト)**
    *   一時変数を保持する Key-Value ストア。
    *   **システム変数**: `$source` (発動カード), `$player` (発動者), `$prev` (直前の結果)。
    *   **ユーザー変数**: `$targets`, `$count` 等、任意の名前で定義可能。

3.  **Pipeline Executor (実行エンジン)**
    *   命令リストを順次実行し、条件分岐（`IF`）やループ（`FOREACH`）を制御するVM。

### 8.3 データ定義例 (JSON Example)

「自分のシールドを1枚墓地に置き、その枚数（1枚）だけ相手クリーチャーを破壊する」効果の例。

```json
"effects": [
  {
    // Step 1: 自分のシールドを1枚選択し、$my_shield に保存
    "op": "SELECT",
    "args": { "zone": "SHIELD_ZONE", "owner": "SELF", "count": 1 },
    "out": "$my_shield"
  },
  {
    // Step 2: $my_shield を墓地へ移動し、成功したカードを $moved に保存
    "op": "MOVE",
    "args": { "cards": "$my_shield", "to": "GRAVEYARD" },
    "out": "$moved"
  },
  {
    // Step 3: 移動できた枚数を $count に保存
    "op": "COUNT",
    "args": { "target": "$moved" },
    "out": "$count"
  },
  {
    // Step 4: $count > 0 なら相手獣を選んで破壊
    "op": "IF",
    "args": {
      "condition": { "op": "GT", "left": "$count", "right": 0 },
      "then": [
        { "op": "SELECT", "args": { "zone": "BATTLE", "owner": "OPPONENT", "count": "$count" }, "out": "$enemy" },
        { "op": "MOVE", "args": { "cards": "$enemy", "to": "GRAVEYARD" } }
      ]
    }
  }
]
```

---

## 9. 移行と互換性戦略 (Migration & Comparison Strategy)

Phase 6への移行において、以下の部分を変更しました。

| 項目 | 現在の実装 (Phase 6 Completed) |
| :--- | :--- |
| **トリガー検知** | `TriggerManager` によるイベント監視 |
| **処理実体** | `GameCommand` (Primitives) |
| **アクション** | `MoveCardHandler` 等が内部で `TransitionCommand` を発行 |

---

## 10. 将来的な理想アーキテクチャ案 (Ideal Architecture Proposal)

（変更なし：Phase 7以降の検討課題として維持）

### 10.1 データ構造の刷新：多態性を持つ命令ツリー (Polymorphic Instruction Tree)
### 10.2 引数管理の刷新：式評価システム (Expression System)
### 10.3 イベント駆動モデルへの完全移行 (Unified Event Listener Model)
### 10.4 エンジンコアの刷新：スタックマシン型VM (Stack Machine VM)
