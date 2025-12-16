# 21. 改定ロードマップ: ハイブリッドエンジンと刷新 (Hybrid Engine Roadmap)

## 概要 (Overview)
本ドキュメントは、「要件定義書 00 (Requirement Definition 00)」の重要変更（Strategic Shift）に基づき、ロードマップを再制定したものである。
旧来の「Legacy JSON Adapter」によるデータ変換戦略を破棄し、「新旧エンジンの共存（ハイブリッド構成）」と「完全なコマンドシステムへの移行」を最優先事項とする。

これにより、プロジェクトのフェーズ定義を以下のように再構築する。

---

## Phase 1-5: 凍結とアーカイブ (Frozen & Archived)
**ステータス: [Frozen]**

以下の領域は、エンジン基盤（Phase 6/7）が安定するまで開発を一時凍結する。

*   **GUI Card Editor (Phase 2/5)**: エンジンのデータ構造（CommandDef）が確定するまで更新を停止。
*   **AI Training Loop (Phase 3/4)**: エンジンのロジック刷新完了後、新アーキテクチャに合わせて再構築を行うため停止。

---

## Phase 6: エンジン刷新 (Engine Overhaul)
**優先度: Critical [Status: WIP]**
**目的**: 柔軟性に欠ける `EffectResolver` を解体し、イベント駆動型アーキテクチャと命令パイプラインへ移行する。

### Task 6.1: イベント駆動基盤 (Event-Driven Foundation) [Done]
*   `TriggerManager` の実装: シングルトン/コンポーネントによるイベント監視システム。
*   `GenericCardSystem` との連携強化。

### Task 6.2: 命令パイプライン (Instruction Pipeline) [Done]
*   `PipelineExecutor` (VM) の実装。
*   アクションをアトミックな命令列として実行する基盤の確立。

### Task 6.3: GameCommand 統合 (GameCommand Integration) [Done]
*   `GameCommand` (Transition, Mutate, Flow) の実装。
*   `GameState::event_dispatcher` とコマンド実行の連携。

---

## Phase 7: ハイブリッドエンジン基盤 (Hybrid Engine Foundation)
**優先度: High [Status: WIP]**
**目的**: 旧エンジン（Macro Actions）と新エンジン（Primitive Commands）を共存させ、段階的な移行を可能にする「ハイブリッド・アーキテクチャ」を構築する。

### Task 7.1: ハイブリッドスキーマの確立 (Hybrid Schema) [Done]
*   JSONデータ構造において、旧来の `actions` リストと新しい `commands` リストの共存を許容するスキーマ変更。
*   `LegacyJsonAdapter` の完全廃止（アダプター開発工数の削減）。

### Task 7.2: デュアルディスパッチの実装 (Dual Dispatch System) [Done/WIP]
*   `GenericCardSystem` を改修し、1枚のカード定義内で `EffectResolver` (Legacy) と `CommandSystem` (New) の両方を順次実行可能にする。
*   これにより、複雑なカードのみ先行してCommand化し、単純なカードはLegacyのまま維持する運用を可能にする。

### Task 7.3: コマンドシステムの実装 (CommandSystem Implementation) [WIP]
*   `CommandSystem` クラスの完全実装。
*   `TRANSITION` (ゾーン移動)、`MUTATE` (状態変更) コマンドのバグ修正と安定化。
*   Pythonバインディング経由でのコマンド実行テストの確立。

---

## Phase 8: AI高度化 - Transformer拡張 (AI Evolution)
**優先度: Future**
**目的**: エンジン刷新完了後、AIモデルを強化する。

### Task 8.1: ハイブリッド埋め込み (Hybrid Embedding)
*   **ID Embedding + Manual Features**: 未知のカード（Zero-shot）に対応するため、カードIDだけでなく、スペック情報（コスト、文明、パワー、効果タグ）をベクトル化して入力する。

### Task 8.2: 文脈拡張 (Context Expansion)
*   Transformerの入力コンテキストに「墓地 (Graveyard)」のカードを含め、墓地利用デッキや探索（Search）の精度を向上させる。

---

## 開発方針の変更点 (Strategic Changes)
1.  **Adapterの廃止**: 旧データを新データに変換するコードは書かない。新旧両方のデータを直接解釈できるエンジンを作る。
2.  **Engine First**: エディタやAIの利便性向上よりも、エンジンの「正しさ」と「拡張性」を最優先する。
3.  **Command Migration**: 最終的には全てのカードロジックを `Command` に置き換えるが、Phase 7の間は共存を許容する。
