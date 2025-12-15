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
    *   **Status**: `TriggerManager`, `GameEvent` クラスの実装とPythonバインディングが完了 (Phase 6.1 Started)。
2.  **GameCommand (Primitives) の実装**
    *   全てのアクションを `TRANSITION`, `MUTATE`, `FLOW`, `QUERY`, `DECIDE` に分解・再実装。
    *   **Status**: `GameCommand` 基底クラスおよび5つのPrimitiveクラスの実装とPythonバインディングが完了 (Phase 6.2 Completed).
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

### 5.2 詳細要件 (Detailed Requirements)

1.  **Event Monitor (B案: イベント監視型)**
    *   **状態トリガーの扱い**: ポーリング（常時監視）ではなく、状態変化イベント（例: `CREATURE_ZONE_ENTER`, `MANA_ZONE_LEAVE`）を監視する方式を採用する。
    *   **理由**: MCTS等のシミュレーション速度を最大化するため、無駄なチェック処理を排除する。
    *   **実装**: エンジンは `COUNT_CHANGED` や `STATE_CONDITION_MET` などの抽象化されたイベントを発行する責務を持つ。

2.  **Loop Prevention (ループ防止)**
    *   **仕様**: トリガーが無限連鎖する場合（A→B→A...）、スタック深度または同一ターン内の発動回数制限により強制停止する。
    *   **実装**: `PendingEffect` に `chain_depth` カウンタを持たせ、閾値（例: 20）を超えたら解決を失敗（Fizzle）させる。

3.  **Context Reference (コンテキスト参照)**
    *   **仕様**: 「破壊されたクリーチャーのパワー以下のクリーチャーを破壊する」のように、イベントの文脈データ（破壊されたカードの情報）を動的に参照する機能。
    *   **実装**: `FilterDef` に `power_max_ref: "EVENT_CONTEXT_POWER"` のような動的参照キーを定義可能にする。

---

## 6. イベント駆動型アクション・リアクション詳細要件 (Event-Driven Action/Reaction Specs)

ターンプレイヤーの権限外で発生するアクション（S・トリガー、革命チェンジ、ニンジャ・ストライク等）の制御仕様。

### 6.1 基本フロー (Basic Flow)
イベント発生 -> トリガー検知 -> 保留効果(PendingEffect)生成 -> 優先権に基づく解決。

### 6.2 詳細要件 (Detailed Requirements)

1.  **Reaction Window (A案: 非同期・ステートマシン型)**
    *   **仕様**: リアクション待機（ニンジャ・ストライク宣言等）が発生した際、エンジンは一時停止（Block）するのではなく、「入力待ち状態（Awaiting Input）」へ遷移し、制御を呼び出し元へ返す。
    *   **理由**: AI（Gym/PettingZoo）との親和性確保のため。AIは観測（Observation）として「入力要求」を受け取り、次のステップで回答（Action）を返す標準的なループで処理できる。
    *   **実装**: `GameState` に `waiting_for_reaction` フラグと `reaction_context` を持たせる。

2.  **Interceptor Layer (置換効果レイヤー)**
    *   **仕様**: 「破壊される代わりに〜する」といった置換効果は、通常のトリガー（事後処理）とは区別し、アクション実行直前に介入する **Interceptor** として実装する。
    *   **フロー**: `ActionGenerator` -> `Interceptor Check` (Modify/Cancel Action) -> `Execute Action` -> `Trigger Event`.

3.  **Optional vs Mandatory (任意と強制)**
    *   **仕様**: リアクションウィンドウにおいて、キャンセル可能か強制発動かを定義。
    *   **実装**: JSON定義に `optional: true/false` を持たせ、強制の場合はUIでキャンセルボタンを無効化、あるいは自動解決する。

---

## 7. GameCommand アーキテクチャ詳細設計 (GameCommand Architecture Specs)

AIとエンジンの共通言語となる「5つの基本命令」の仕様詳細。

### 7.1 5つの基本命令 (Primitives)

1.  **TRANSITION**: カードの移動（ゾーン間、状態変更）。
2.  **MUTATE**: カード/プレイヤーのプロパティ変更（パワー修正、フラグ付与）。
3.  **FLOW**: ゲーム進行の制御（フェーズ遷移、ステップ移行）。
4.  **QUERY**: エンジンから外部（AI/UI）への選択要求。
5.  **DECIDE**: 外部からの選択結果の適用。

### 7.2 詳細要件 (Detailed Requirements)

1.  **DECIDE Command (A案: 回答/適用型)**
    *   **定義**: `DECIDE` は「AI/プレイヤーによる意思決定の結果（回答）」として定義する。エンジンが発行するものではなく、外部からエンジンへ投入される確定情報である。
    *   **データ**: `target_index`, `card_id`, `option_id` 等の具体的な選択内容のみを保持する。
    *   **リプレイ性**: ログには `DECIDE` のみが記録され、再生時はエンジンが内部生成した `QUERY` に対してログの `DECIDE` を適用することで再現を行う。

2.  **FLOW Granularity (A案: 詳細粒度)**
    *   **仕様**: `FLOW` コマンドはフェーズだけでなく、効果解決のステップ（Step）や処理の区切り（Micro-step）単位で発行する。
    *   **理由**: イベント駆動システムのフックポイントを明確にし、複雑な処理の途中状態を透明化するため。

3.  **Rollback Support (Must Have: 内蔵型)**
    *   **要件**: MCTS探索の高速化（コピー負荷削減）のため、GameCommand層に **Undo（逆操作）** 機能を内蔵することを必須とする。
    *   **実装**: 各コマンドクラスは `execute()` と対になる `invert()` メソッド、または逆操作コマンド生成機能を持ち、O(1)〜O(Δ)コストで状態を復元可能にする。

---

## 8. 命令パイプラインと汎用アクション構造 (Instruction Pipeline & Generalized Action Structure)

ハードコード（複合効果の個別C++実装）を撤廃し、あらゆるカード効果をデータ定義（JSON）のみで実現するための新アーキテクチャ。

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

### 8.4 メリット
*   **汎用性**: C++の修正なしに、新しいロジック（変数を介した複雑な連動）をJSONのみで記述可能。
*   **ステートフル**: 「さっき破壊したカードのコスト」等の文脈情報を変数として保持・参照できる。
*   **割り込み耐性**: パイプラインの実行位置（Program Counter）とContextを保存すれば、S・トリガー等の割り込み後も正確に復帰できる。

---

## 9. 移行と互換性戦略 (Migration & Comparison Strategy)

Phase 6への移行において、どの部分を変更し、どの部分を維持するかを明確にする。

### 9.1 変更・廃止する部分 (To Change / Deprecate)

| 項目 | 現在の実装 (Phase 0-4) | 今後の実装 (Phase 6) | 変更理由 |
| :--- | :--- | :--- | :--- |
| **トリガー検知** | ハードコードされたフックポイント | `TriggerManager` によるイベント監視 | 拡張性とスパゲッティコード解消 |
| **処理実体** | `EffectResolver` の巨大な `switch` 文 | `Instruction Executor` (VM) | 組み合わせ爆発への対応 |
| **データ受け渡し** | 限定的な `execution_context` | 完全な変数システム (`Context`) | 柔軟なロジック記述のため |
| **中断・再開** | 関数コールスタック依存 | PC (Program Counter) 保存 | ロールバック機能実現のため |

### 9.2 流用・継続する部分 (To Keep / Reuse)

| 項目 | 理由 (Why keep it?) |
| :--- | :--- |
| **JSONデータ構造** | `CardData`, `FilterDef` 等の定義は、エディタ資産および学習済みAIとの互換性維持のため、可能な限り維持する。新しいエンジンはこれらのデータを読み込み、内部的にGameCommandへ変換する。 |
| **TriggerType** | `ON_PLAY`, `ON_ATTACK` などの概念自体は不変であり、イベント名としてマッピングして利用する。 |
| **ConditionDef** | フィルタ条件の定義構造 (`type`, `value`, `op`) は、イベントフィルタとしてそのまま有用。 |
| **CostPaymentSystem** | 独立性が高く、GameCommand化の影響を受けにくいため、コンポーネントとして再利用する。 |

---

## 10. 将来的な理想アーキテクチャ案 (Ideal Architecture Proposal)

「既存の資産（エディタ・AI・JSON形式）との互換性維持」という制約を撤廃し、エンジンの表現力と拡張性を最大化する場合に採用すべき「完全データ駆動型」の設計案。
将来的に本アーキテクチャへ移行する場合は、既存のJSONデータを新形式へ変換する **トランスパイラ (Converter)** を開発することで、資産の互換性を担保する。

### 10.1 データ構造の刷新：多態性を持つ命令ツリー (Polymorphic Instruction Tree)

「巨大な構造体（Fat Struct）」を廃止し、命令の種類ごとに最適化されたスキーマを持つ構造へ移行する。

*   **現状**: `ActionDef` に全てのパラメータ（数値、文字列、フィルタ、サブアクション等）が含まれており、メモリ効率が悪く拡張性が低い。
*   **提案**: 基底クラス `Instruction` を継承した、目的別の型定義を採用する。JSON上では `op` コードによるタグ付きユニオンとして扱う。

```json
// 例: 「マナゾーンから1枚手札に戻す」
{
  "op": "MOVE_CARD",
  "source": { "zone": "MANA", "owner": "SELF" },
  "destination": { "zone": "HAND" },
  "count": 1,
  "select_strategy": "MANUAL"
}
```

### 10.2 引数管理の刷新：式評価システム (Expression System)

単純なKey-Value参照を廃止し、ネスト可能な **式 (Expression)** オブジェクトへ汎用化する。

*   **現状**: `value: 5000` (定数) または `value_key: "power_val"` (単純変数参照) のみ。
*   **提案**: 全ての数値・文字列引数を `Expression` 型とし、エンジン内での動的な計算を可能にする。

```json
// 例: 「自分のシールド枚数 × 1000」のパワーを追加
{
  "op": "MODIFY_POWER",
  "value": {
    "op": "MULTIPLY",
    "args": [
      { "op": "COUNT", "target": { "zone": "SHIELD", "owner": "SELF" } },
      1000
    ]
  }
}
```

### 10.3 イベント駆動モデルへの完全移行 (Unified Event Listener Model)

`CardKeywords` (`cip`, `slayer` 等) や `TriggerType` 列挙型によるハードコードされた分岐を全廃し、全てを **イベントリスナー** として統一する。

*   **現状**: `if (card.keywords.slayer) ...` のような条件分岐がエンジン内に散在している。
*   **提案**: カードは「属性」ではなく、「反応するイベント」と「実行する命令」のリストを持つ。
    *   **CIP**: `event: "ZONE_ENTER", destination: "BATTLE_ZONE"` に対するリスナー。
    *   **スレイヤー**: `event: "BATTLE_LOSE"` に対するリスナー。
    *   **独自トリガー**: 「相手が呪文を唱えた時」なども、エンジンの改造なしにJSON定義のみで追加可能になる。

### 10.4 エンジンコアの刷新：スタックマシン型VM (Stack Machine VM)

再帰的な関数呼び出し (`resolve_effect` -> `resolve_action`) を廃止し、**命令キューを持つスタックマシン** としてエンジンを実装する。

*   **メリット**:
    *   **中断・再開**: ユーザー入力待ちや非同期処理において、プログラムカウンタ (PC) とスタック状態を保持したまま停止 (`YIELD`) できる。
    *   **複雑な割り込み**: S・トリガーや割り込み効果を、命令キューへの動的な挿入として統一的に処理できる。
