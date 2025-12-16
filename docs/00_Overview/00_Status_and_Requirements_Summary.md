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
*   **EffectResolver (Legacy)**: 現在の主力ロジック。巨大なswitch文により効果処理を行っているが、複雑化により限界に達している。Phase 6で廃止予定。
*   **GameCommand (Partial)**: 基本クラスと一部のコマンドは実装済みだが、エンジンの中核ロジックとしては未統合。
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
    *   従来のハードコードされたトリガーフックを、イベント発行 (`dispatch`) に置き換える。
    *   **New**: `check_triggers` メソッドによる「Passive -> Triggered -> Interceptor」の順次チェック機構を実装済み。
*   **Step 2: 命令パイプライン (Instruction Pipeline) の実装**
    *   JSON定義された命令列を実行する `PipelineExecutor` (VM) の実装。（基盤実装完了。命令ハンドラの拡張が必要）
    *   `EffectResolver` の各ロジックを `Instruction` (Move, Modify, Check等) の組み合わせに分解・再実装。
    *   **New**: `PipelineExecutor` の命令ハンドラ (`SELECT` with `TargetUtils`, `MODIFY`, `LOOP`) を実装済み。
    *   **New**: `LegacyJsonAdapter` を実装済み。従来のJSONデータを `Instruction` 列に変換可能になった。
        *   基本アクション (`DRAW`, `MOVE`, `DESTROY`, `SEARCH_DECK`) の検証完了。
        *   `PipelineExecutor` のゾーン参照をO(1)に最適化済み。
*   **Step 3: GameCommand への統合**
    *   **Status: Completed**
    *   全てのアクションを `GameCommand` (Transition, Mutate, Flow等) 発行として統一し、Undo/Redo基盤を確立する。
    *   **Implemented**: `GameState` に `execute_command` と `command_history` を実装。
    *   **Implemented**: `PhaseManager` (フェーズ遷移、ターン開始、ドロー)、`ManaSystem` (マナタップ/アンタップ)、`PipelineExecutor` (カード移動、変異) を GameCommand 化。
    *   **New**: `PendingEffect` を独立ヘッダに分離し、`ADD_PENDING_EFFECT`, `SET_ACTIVE_PLAYER` コマンドを追加することで、複雑なターン遷移（フェーズ変更→トリガー追加→プレイヤー交代）の完全な Undo/Redo を実現。
*   **Step 4: 移行と検証**
    *   既存テストケースの新エンジン上でのパス確認。

### 3.2 [Pending] Phase 3.2: AI 本番運用 (Production Run)
**Status: On Hold (Waiting for Phase 6)**
エンジン刷新完了後、新アーキテクチャ上でAI学習を再開します。
*   GameCommand化による「Undo」機能を活用し、MCTS探索速度の向上を見込む。

### 3.3 [Frozen] Phase 5: エディタ機能の完成 (Editor Polish)
**Status: Frozen**
エンジン刷新完了後、必要に応じてデータ構造の変更をエディタに反映させます。

---

## 4. 汎用コストおよび支払いシステム (General Cost and Payment System)

（変更なし。Phase 6においても `CostPaymentSystem` は継続利用する。）

---

## 5. イベント駆動型トリガーシステム詳細要件 (Event-Driven Trigger System Specs)

**目的**: `EffectResolver` 内の分散したトリガーチェック処理を、一元管理されたイベントシステムへ置換する。

### 5.1 アーキテクチャ構成 (Architecture)

1.  **EventObject (イベント定義)**
    *   不変 (Immutable) なデータ構造。
    *   **Fields**:
        *   `type`: イベント種別 (`EventType::ZONE_ENTER`, `EventType::ATTACK_INIT` 等)
        *   `source_id`: 発生源のカードID
        *   `player_id`: 発生させたプレイヤー
        *   `context`: 詳細情報を持つ Map (`target_id`, `cost_paid` 等)

2.  **TriggerManager (イベント管理)**
    *   **責務**: 全イベントの集約とリスナーへの配信。
    *   **Method `dispatch(EventObject)`**:
        1.  常時監視効果 (Passive Effects) のチェック。
        2.  誘発型能力 (Triggered Abilities) の検索 (`PendingEffect` 生成)。
        3.  置換効果 (Interceptor) の適用確認。

3.  **Listener Registration (登録)**
    *   カード実体 (`CardInstance`) ではなく、カード定義 (`CardDefinition`) または現在のゲーム状態がリスナーを持つ。
    *   最適化のため、各ゾーンのカードがどのイベントを購読しているかをキャッシュする仕組みを検討する。

### 5.2 実装要件

*   **廃止**: `resolve_trigger` 関数内の switch 分岐。
*   **導入**: `TriggerManager::check_triggers(event)` メソッド。
*   **データバインディング**: JSONの `TriggerType` 文字列を `EventType` enum にマッピングする変換層を設ける。

---

## 6. イベント駆動型アクション・リアクション詳細要件 (Event-Driven Action/Reaction Specs)

**目的**: ターンプレイヤー以外の行動（S・トリガー、ニンジャ・ストライク）を、エンジンのメインループ外の特例処理ではなく、ステートマシンの一部として正規化する。

### 6.1 リアクションウィンドウ (Reaction Window)

*   **Awaiting Input State**:
    *   リアクション可能なタイミング（攻撃時、ブロック時、シールドブレイク時）で、エンジンは一時停止状態 (`GameState::Status::WAITING_FOR_REACTION`) に遷移する。
    *   この状態では、優先権を持つプレイヤー（非ターンプレイヤーの場合もある）からの `DECLARE_REACTION` または `PASS` コマンドのみを受け付ける。

*   **処理フロー**:
    1.  イベント発生 (`ATTACK_INIT` 等)
    2.  `TriggerManager` がリアクション可能なカード（ニンジャ・ストライク持ち等）を検知。
    3.  候補が存在する場合、`ReactionWindow` オブジェクトを作成し、スタックに積む。
    4.  ゲーム状態を `WAITING` に変更。
    5.  外部エージェントが行動を選択。
    6.  全員がパスするまでウィンドウを維持し、終了後に元の処理へ戻る。

### 6.2 インターセプター (Interceptor / Replacement Effects)

*   **定義**: イベントの発生自体を書き換える効果（「破壊される代わりに手札に戻る」等）。
*   **実装**:
    *   アクション実行前に `ActionGenerator` が `TriggerManager::check_interceptors(action)` をコール。
    *   置換効果が存在する場合、元のアクションを破棄し、置換後のアクションを実行する。

---

## 7. GameCommand アーキテクチャ詳細設計 (GameCommand Architecture Specs)

**目的**: 全ての状態変更操作を「コマンド」としてカプセル化し、Undo/Redo とログ記録を統一する。

### 7.1 基本命令セット (Primitives)

以下の5つのコマンドクラスですべてのゲームロジックを表現する。

1.  **CMD_TRANSITION (移動)**:
    *   カードのゾーン移動。
    *   `args`: `card_id`, `source_zone`, `dest_zone`, `index`.
2.  **CMD_MUTATE (状態変更)**:
    *   パワー変更、タップ/アンタップ、シールド化、効果付与。
    *   `args`: `target_id`, `property`, `value`, `duration`.
3.  **CMD_FLOW (制御)**:
    *   フェーズ遷移、ターン終了、ステップ移行。
4.  **CMD_QUERY (要求)**:
    *   エンジンからエージェントへの入力要求（対象選択など）。
    *   これはエンジン内部状態を変更せず、UI/AIへのシグナルとして機能する。
5.  **CMD_DECIDE (決定)**:
    *   エージェントからの回答。
    *   `CMD_QUERY` に対する応答として処理され、確定した選択内容をコンテキストに書き込む。

### 7.2 移行要件

*   `GameState` の `vector` や `map` を直接操作するメソッド（`add_card_to_hand` 等）は、原則として `GameCommand` 経由でのみ呼び出されるようにリファクタリングする（または private 化する）。
*   各コマンドは `execute()` と `undo()` を実装し、MCTS探索時の状態復元を高速に行えるようにする。

---

## 8. 命令パイプライン (Instruction Pipeline) 詳細要件

**目的**: `EffectResolver` のハードコードされた効果ロジックを、JSON定義可能な「命令」のリスト実行へ置き換える。

### 8.1 構造

*   **PipelineExecutor**:
    *   命令リスト (`List<Instruction>`) を受け取り、順次実行するクラス。
    *   現在の実行位置 (PC) を管理し、`CMD_QUERY` 発生時には実行を中断、`CMD_DECIDE` 受領後に再開する機能を持つ。

*   **Context (実行コンテキスト)**:
    *   変数スコープ。
    *   システム変数: `$source`, `$player`, `$event_context`.
    *   ローカル変数: 命令間で受け渡されるデータ（選択されたカードIDリスト等）。

### 8.2 Instruction (命令) の種類

既存の `EffectActionType` を、より粒度の細かい命令に分解する。

1.  **Logic Instructions**:
    *   `IF`, `ELSE`, `LOOP (FOREACH)`: 制御構文。
2.  **Query Instructions**:
    *   `SELECT`: プレイヤーに対象選択を要求する（`CMD_QUERY` を発行）。
3.  **Action Instructions**:
    *   `MOVE`: `CMD_TRANSITION` を発行。
    *   `MODIFY`: `CMD_MUTATE` を発行。
4.  **Calculation Instructions**:
    *   `COUNT`: ゾーンのカード枚数などを数え、変数に格納。
    *   `MATH`: 数値演算。

### 8.3 データ定義例

```json
"effects": [
  { "op": "SELECT", "filter": { "zone": "BATTLE", "owner": "OPPONENT" }, "count": 1, "out": "$target" },
  { "op": "IF", "cond": { "exists": "$target" }, "then": [
      { "op": "MOVE", "target": "$target", "to": "HAND" }
    ]
  }
]
```

---

## 9. 移行戦略 (Migration Strategy)

**Phase 6 完遂のための具体的ステップ**

1.  **基盤実装**: `TriggerManager`, `PipelineExecutor` クラスのC++実装。
2.  **ラッパー作成**: 既存のJSONデータを読み込み、ランタイムで新しい `Instruction` 形式に変換するアダプター (`LegacyJsonAdapter`) を作成する。これにより、エディタやJSONファイルの即時書き換えを回避する。（実装完了）
3.  **段階的置換**:
    *   まず単純な効果（W・ブレイカー、ブロッカー等）から新システムへ移行。
    *   次に `CIP` (出た時) 効果のパイプライン処理化。
    *   最後に複雑な効果（S・トリガー、革命チェンジ）を移行。
4.  **EffectResolverの廃止**: 全ロジックの移行完了後、`EffectResolver.cpp` を削除する。

---

## 10. 将来的な理想アーキテクチャ案 (Future Scope)

**Note: 本セクションは Phase 6 のスコープ外です。**
スタックマシン型VMや完全なバイトコード化などの高度な最適化は、Phase 6 の機能が安定し、AI学習が軌道に乗った後の「Phase 7以降」の課題とします。現在は考慮しません。

