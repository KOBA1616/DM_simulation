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

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Card Editor Ver 2.3**: 3ペイン構成（ツリー/プロパティ/プレビュー）。
    *   **テキスト生成**: 数値範囲や任意選択の日本語生成ロジック強化済み。
    *   **リアクション編集**: `ReactionWidget` による動的UI切り替えをサポート。
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
*   現在、主要な不整合は解消されました。

### 2.6 現在の懸念事項と既知の不具合 (Current Concerns and Known Issues)
*   特になし。

※ 完了した詳細な実装タスクは `docs/00_Overview/99_Completed_Tasks_Archive.md` にアーカイブされています。

---

## 3. 詳細な開発ロードマップ (Detailed Roadmap)

エンジンの根本的な刷新（GameCommand化）を優先し、その後にエディタやAI本番運用を進める方針でロードマップを再編しました。

### 3.0 [Priority: High] Phase 6: GameCommand アーキテクチャとエンジン刷新 (Engine Overhaul)

AI学習効率と拡張性を最大化するため、エンジンのコアロジックを「イベント駆動型」かつ「5つの基本命令 (GameCommand)」に基づくアーキテクチャへ刷新します。
メテオバーン、超次元、その他未実装の特殊メカニクスは、このフェーズにおける「アクションの汎用化」によって自然に解決・実装されます。

1.  **イベント駆動型トリガーシステムの実装**
    *   ハードコードされたフックポイントを廃止し、`TriggerManager` による一元管理へ移行。
2.  **GameCommand (Primitives) の実装**
    *   全てのアクションを `TRANSITION`, `MUTATE`, `FLOW`, `QUERY`, `DECIDE` に分解・再実装。
3.  **アクション汎用化**
    *   `MOVE_CARD` の完全統合、`APPLY_MODIFIER` の汎用化により、個別実装を排除。

### 3.1 [Priority: Medium] Phase 5: エディタ機能の完成 (Editor Polish & Validation)

エンジン刷新後、新しいデータ構造に合わせてエディタのバリデーションを強化します。

1.  **Logic Mask (バリデーション) の実装**
    *   公式ルールに基づく最小限のマスク処理を実装。過度な制限は設けず、明らかな矛盾のみを防ぐ。
    *   **ルール**:
        *   **呪文 (Spell)**: 「パワー」フィールドを無効化（0固定）。
        *   **進化クリーチャー**: 「進化条件」の設定を有効化。
        *   **その他**: 基本的に制限なし（ユーザーの自由度を確保）。

### 3.2 [Priority: Post-Phase 6] AI 本番運用 (Production Run)

「最強のAI」を目指す大規模学習は、GameCommandアーキテクチャの完成後に行います。

*   **タイミング**: Phase 6 完了後。
*   **理由**: GameCommandによるアクション空間の圧縮（意味ベースの学習）が完了した状態で学習させることで、未知のカードへの汎化性能と学習効率が劇的に向上するため。
*   **現在**: Phase 4完了時点のモデルを用いた小規模なパイロット運用（技術検証）に留める。

---

## 4. 汎用コストおよび支払いシステム (General Cost and Payment System)

（変更なし）

---

## 5. イベント駆動型トリガーシステム詳細要件 (Event-Driven Trigger System Specs)

既存の `EffectResolver` を刷新するための技術要件。以下の不足項目を補完して実装すること。

### 5.1 基本アーキテクチャ (Architecture)
*   **TriggerManager**: 全イベントの発行 (`dispatch`) と購読 (`subscribe`) を管理するシングルトン/コンポーネント。
*   **Event Object**: `type`, `source`, `target`, `context` (Map) を持つ不変オブジェクト。

### 5.2 詳細要件と不足事項 (Detailed Requirements & Missing Gaps)

1.  **State Triggers vs Event Triggers (状態トリガーとイベントトリガーの区別)**
    *   **イベントトリガー**: 「〜した時」 (When/Whenever)。イベント発生の一瞬だけ誘発する。
    *   **状態トリガー**: 「〜である限り」や条件達成時 (When state meets condition)。
    *   **要件**: `TriggerManager` はイベントだけでなく、状態チェック（State Check）のポーリングもサポートするか、あるいは「状態変化イベント」として抽象化する必要がある（例: シールドが0枚になった瞬間に `SHIELD_ZERO` イベントを発行する）。

2.  **Intervening 'If' Clause (割り込み条件判定)**
    *   **仕様**: 「〜した時、もし〜なら、XXする」という構文において、トリガー時点だけでなく **解決時点** にも条件を満たしているかチェックする機能。
    *   **実装**: `TriggerDef` に `condition` (トリガー時) と `resolution_condition` (解決時) の2つのフィルタを持たせる。

3.  **Context Reference (コンテキスト参照)**
    *   **仕様**: 「破壊されたクリーチャーのパワー以下のクリーチャーを破壊する」のように、イベントの文脈データ（破壊されたカードの情報）を動的に参照する機能。
    *   **実装**: `FilterDef` に `power_max_ref: "EVENT_CONTEXT_POWER"` のような動的参照キーを定義可能にする。

---

## 6. イベント駆動型アクション・リアクション詳細要件 (Event-Driven Action/Reaction Specs)

ターンプレイヤーの権限外で発生するアクション（S・トリガー、革命チェンジ、ニンジャ・ストライク等）の制御仕様。

### 6.1 基本フロー (Basic Flow)
イベント発生 -> トリガー検知 -> 保留効果(PendingEffect)生成 -> 優先権に基づく解決。

### 6.2 詳細要件と不足事項 (Detailed Requirements & Missing Gaps)

1.  **APNAP (Active Player, Non-Active Player) 解決順序**
    *   **仕様**: ターンプレイヤー(AP)と非ターンプレイヤー(NAP)のトリガーが同時に誘発した場合、**APの処理が全て終わってからNAPの処理を行う**（あるいはスタックへの積み順を制御する）。
    *   **実装**: `TriggerManager` はトリガーを即時実行せず、プレイヤーごとの「待機リスト」に入れ、AP -> NAP の順でスタックにプッシュするロジックを実装する。

2.  **Batch Processing (バッチ処理)**
    *   **仕様**: 「クリーチャーが3体同時に破壊された時」のようなイベントに対し、トリガーを1回だけ発火させるか、3回発火させるかの制御。
    *   **実装**: `Event` に `batch_id` または `is_simultaneous` フラグを持たせ、トリガー定義側で `batch_mode: "ONCE_PER_BATCH" | "PER_INSTANCE"` を指定可能にする。

3.  **Optional vs Mandatory (任意と強制)**
    *   **仕様**: リアクションウィンドウ（ニンジャ・ストライク宣言画面など）において、キャンセル可能か強制発動かを定義。
    *   **実装**: JSON定義に `optional: true/false` を持たせ、強制の場合はUIでキャンセルボタンを無効化、あるいは自動解決する。

---

## 7. GameCommand アーキテクチャ詳細設計 (GameCommand Architecture Specs)

AIとエンジンの共通言語となる「5つの基本命令」の仕様詳細。

### 7.1 5つの基本命令 (Primitives)
`TRANSITION`, `MUTATE`, `FLOW`, `QUERY`, `DECIDE`。

### 7.2 詳細要件と不足事項 (Detailed Requirements & Missing Gaps)

1.  **Undo/Rollback Support (巻き戻し機能)**
    *   **重要**: MCTS等の探索アルゴリズムにおいて、状態のコピー（Clone）は重いため、アクションの逆操作（Undo）による高速な状態復帰が求められる。
    *   **要件**: 各GameCommandは `invert()` メソッド、または逆操作コマンド（例: `TRANSITION` の逆は移動元へ戻す）を生成・実行可能でなければならない。
    *   **実装**: `GameCommand` 実行時に「変更前の値（Snapshot）」を記録し、Undo時にそれを復元する仕組みを組み込む。

2.  **Source & Log Metadata (ソース情報とログ)**
    *   **重要**: 「5つの命令」だけでは「なぜその移動が起きたか（ボルメテウスの効果か、ブロックの結果か）」が分からなくなる。
    *   **要件**: `GameCommand` 構造体に `source_id` (発生源のカード/効果ID) と `reason` (理由コード)、および人間可読な `log_message` を含める。

3.  **Serialization (シリアライズ)**
    *   **重要**: リプレイ保存や通信対戦のため、一連のコマンド列を軽量なバイナリまたはテキスト形式で保存・復元できること。
    *   **実装**: Protobuf または独自の軽量バイナリフォーマットを策定し、`GameCommand` 列の完全な再現性を保証する。

4.  **Action Generalization (アクション汎用化のゴール)**
    *   `DESTROY`, `MANA_CHARGE` などの既存アクションは、最終的にすべて `TRANSITION` コマンドを発行する「ラッパー（マクロ）」として再定義される。これにより、将来的に「超次元ゾーンへの移動」などが増えても、`TRANSITION` のパラメータを変えるだけで対応完了とする。
