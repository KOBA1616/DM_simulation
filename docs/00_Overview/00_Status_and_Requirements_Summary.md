# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

現在、**Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** を最優先事項として進行中です。
既存のハードコードされた効果処理 (`EffectResolver`) を廃止し、イベント駆動型アーキテクチャと命令パイプライン (`Instruction Pipeline`) へ刷新することで、柔軟性と拡張性を確保します。

AI学習 (Phase 3) およびエディタ開発 (Phase 5) は、このエンジン刷新が完了するまで一時凍結します。

### 重要変更 (Strategic Shift)
既存のJSONデータ（Legacy JSON）の再利用や変換アダプタ (`LegacyJsonAdapter`) の開発は**完全に放棄・廃止**します。
今後は新エンジン (`CommandSystem` / `PipelineExecutor`) に最適化された新しいJSON形式のみをサポートし、過去の資産に縛られず、エンジンの完成度と品質を最優先します。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   **EffectResolver (Legacy)**: 現在の主力ロジックだが、段階的に廃止予定。
*   **GameCommand**: 新エンジンの核となるコマンドシステム。`Transition`, `Mutate`, `Flow` などのプリミティブを実装済み。
*   **LegacyJsonAdapter**: **削除済み**。旧データサポートは終了。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Status**: 稼働中 (Ver 2.3)。
*   **Freeze**: 新JSONスキーマが確定次第、新フォーマット専用エディタとして改修を行う。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   **Status**: パイプライン構築済み。
*   **Pending**: エンジン刷新に伴うデータ構造の変更が確定するまで凍結。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] Phase 6: エンジン刷新 (Engine Overhaul)
**Status: In Progress**
`EffectResolver` を解体し、イベント駆動型システムと命令パイプラインへ完全移行します。

*   **Step 1: イベント駆動基盤の実装**
    *   **Status: Implemented**
    *   `TriggerManager`: シングルトン/コンポーネントによるイベント監視・発行システムの実装。（実装完了）
    *   `check_triggers` メソッドにより、`GameEvent` をトリガーとして `PendingEffect` を生成するフローを確立。
*   **Step 2: 命令パイプライン (Instruction Pipeline) の実装**
    *   **Status: Implemented & CIP Integrated**
    *   `PipelineExecutor` (VM) を実装済み。
    *   `GenericCardSystem` を更新し、一部のトリガー処理をパイプラインへルーティング可能に。
    *   **Deleted**: `LegacyJsonAdapter` は廃止されました。
*   **Step 3: GameCommand への統合**
    *   **Status: Completed**
    *   全てのアクションを `GameCommand` (Transition, Mutate, Flow等) 発行として統一し、Undo/Redo基盤を確立する。
    *   **New**: `GameInstance` にて `TriggerManager` を `GameState::event_dispatcher` と連携させ、コマンド実行時のイベント発行をトリガー検知につなげる統合を完了。

### 3.2 [Priority: High] Phase 7: ハイブリッド・エンジン基盤 (New Requirement)
**Status: In Progress (Latest)**
旧エンジン（マクロ的アクション）と新エンジン（プリミティブコマンド）を共存・統合させるためのアーキテクチャ実装。

*   **Step 1: データ構造の刷新 (Hybrid Schema)**
    *   **Status: In Progress**
    *   JSONスキーマに `CommandDef` を導入。
    *   **Policy Change**: `EffectDef` に `commands` フィールドを追加するが、旧来の `actions` からの自動変換は行わない。新形式での記述を必須とする。
*   **Step 2: CommandSystem の実装**
    *   **Status: Implemented**
    *   `dm::engine::systems::CommandSystem` を新設。
    *   Pythonバインディングを整備し、外部からのコマンド実行テストが可能になった。

### 3.3 [Priority: Future] Phase 8: Transformer拡張 (Hybrid Embedding)
**Status: Planned**
Transformer方式を高速化し、かつZero-shot（未知のカードへの対応）を可能にするため、「ハイブリッド埋め込み (Hybrid Embedding)」を導入します。また、文脈として墓地のカードも対象に含めます。

*   **コンセプト (Concept)**
    *   **Hybrid Embedding**: `Embedding = Embed(CardID) + Linear(ManualFeatures)`
    *   **Zero-shot対応**: 未知のカード（ID埋め込み未学習）でも、スペック情報（コスト、文明、パワー等）から挙動を推論可能にする。
    *   **学習の高速化**: スペック情報から即座に役割（アタッカー、防御札）を判断し、初期学習の収束を早める。ID埋め込みは長期的なコンボ相性の補完に使用する。
    *   **スコープ拡張**: Transformerの入力文脈に「墓地」のカードも含め、墓地利用や探索に対応させる。

*   **実装要件 (Requirements)**
    *   **A. C++側 (TensorConverter)**
        *   `convert_to_sequence` を修正し、`Output: (TokenSequence, FeatureSequence)` を返すように変更する。
        *   `FeatureSequence`: 各トークンに対応するカードの特徴量ベクトル `vector<vector<float>>`。
    *   **B. Python側 (NetworkV2)**
        *   モデル入力層を修正: `x_id` (Card IDs) と `x_feat` (Manual Features) を受け取る。
        *   `x_feat` を `nn.Linear` で埋め込み次元に変換し、ID埋め込みと加算する。
    *   **C. 特徴量ベクトルの定義 (Feature Vector Definition)**
        *   コスト (Cost)
        *   パワー (Power)
        *   文明 (Civilization)
        *   カードタイプ (Card Type)
        *   キーワード能力: ブロッカー, シールドトリガー, ガードストライク, スピードアタッカー, マッハファイター, n枚ブレイカー
        *   リソース操作 (Move Card): 移動元・移動先（マナ、手札、墓地、シールド）
        *   除去 (Removal): 破壊、バウンス、マナ送り、シールド送り、封印
        *   コスト踏み倒し (Cost Cheating)
        *   コスト軽減 (Cost Reduction)
        *   能力の発動タイミング: cip, atk, pig, 離れた時, ターン開始時, 終了時, 常在効果, 相手のターン中, 自分のターン中
        *   対策 (Meta): クリーチャー対策, 呪文対策, 墓地対策, シールドトリガー対策, コスト踏み倒し対策
        *   殿堂 (Hall of Fame Status)

## 4. 今後の課題 (Future Tasks)

1.  **Primitive Command Execution Fix**:
    *   `CommandSystem` におけるプリミティブ実行 (`TRANSITION`) の不具合修正。
2.  **Full Trigger System Migration**:
    *   `EffectResolver` が現在行っている処理を全て `CommandSystem` 経由に切り替える。
3.  **New JSON Standard Adoption**:
    *   カードデータの新JSON形式への移行推進。Legacyサポートの完全撤廃に伴い、データセットの再構築を行う。
4.  **GUI Update**:
    *   `CardEditor` を更新し、新スキーマ (`CommandDef`) の編集に対応させる。
