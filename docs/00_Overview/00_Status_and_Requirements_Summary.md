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
*   **EffectResolver (Legacy)**: 現在の主力ロジック。巨大なswitch文により効果処理を行っているが、複雑化により限界に達している。そのため、game command方式に移行中。
*   **GameCommand (Partial)**: 基本クラスと一部のコマンドは実装済みだが、エンジンの中核ロジックとしては未統合。
*   **汎用コストシステム**: 実装済み。新エンジンでもそのまま利用する。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   **Status**: 稼働中 (Ver 2.3)。
*   **Freeze**: エンジン刷新に伴うデータ構造の変更が確定するまで、機能追加および改修を凍結する。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   **Status**: パイプライン構築済み。
*   **Pending**: エンジン刷新に伴うデータ構造の変更が確定するまで、機能追加および改修を凍結する。

---

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
    *   **New**: `GenericCardSystem` を更新し、`ON_PLAY` (CIP) トリガー処理を `PipelineExecutor` へルーティング開始。
    *   **New**: `LegacyJsonAdapter` を拡張し、`SEARCH_DECK` などの複雑なアクションを含む既存のJSONデータを命令列に変換可能にした。
    *   **Verify**: `tests/verify_cip_pipeline.py` にて標準的なCIP効果（ブロンズ・アーム・トライブ）および条件付き効果（マナ武装）がパイプライン経由で正しく動作することを確認済み。
*   **Step 3: GameCommand への統合**
    *   **Status: Completed**
    *   全てのアクションを `GameCommand` (Transition, Mutate, Flow等) 発行として統一し、Undo/Redo基盤を確立する。
    *   **New**: `GameInstance` にて `TriggerManager` を `GameState::event_dispatcher` と連携させ、コマンド実行時のイベント発行をトリガー検知につなげる統合を完了。
*   **Step 4: 移行と検証**
    *   **Known Issue**: `python/tests/test_mega_last_burst.py` が `DestroyHandler` の依存関係（CardRegistry vs CardDB）により失敗している。次フェーズで `GenericCardSystem::resolve_action` とハンドラの連携を詳細調査・修正する。
    *   **New**: S・トリガーおよび革命チェンジのイベント駆動フロー（リアクションウィンドウ生成）の検証完了。

### 3.2 [Priority: High] Phase 7: ハイブリッド・エンジン基盤 (New Requirement)
**Status: In Progress (Latest)**
旧エンジン（マクロ的アクション）と新エンジン（プリミティブコマンド）を共存・統合させるためのアーキテクチャ実装。

*   **Step 1: データ構造の刷新 (Hybrid Schema)**
    *   **Status: Completed**
    *   JSONスキーマに `CommandDef` を導入し、マクロ（`DRAW_CARD`）とプリミティブ（`TRANSITION`, `MUTATE`）を混在可能にした。
    *   `EffectDef` に `commands` フィールドを追加し、旧来の `actions` フィールドからの自動変換ロジックを `CardRegistry` に実装済み。既存のJSON資産をそのまま新エンジンで読み込み可能。
*   **Step 2: CommandSystem の実装**
    *   **Status: Implemented**
    *   `dm::engine::systems::CommandSystem` を新設。マクロコマンドを `GameCommand` 列（プリミティブ）に展開して実行するロジックを集約。
    *   Pythonバインディングを整備し、外部からのコマンド実行テストが可能になった。
*   **Step 3: 検証 (Verification)**
    *   **Status: Partially Verified**
    *   `tests/verify_hybrid_engine.py` にて、旧JSONデータ（`DRAW_CARD`）が新エンジンの `CommandSystem` を通じて正しくカード移動処理（Deck->Hand）を行うことを確認済み。
    *   **Issue**: プリミティブコマンド (`TRANSITION`) の直接実行テストにおいて、ターゲット解決後の状態更新（Deck->Mana）が正しく反映されない事象を確認中。`TransitionCommand` または `TargetScope` の解決ロジックに修正が必要。

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
    *   `CommandSystem` におけるプリミティブ実行 (`TRANSITION`) の不具合修正。ターゲット解決 (`resolve_targets`) とコマンド発行の連携を見直す。
2.  **Full Trigger System Migration**:
    *   `EffectResolver` が現在行っている処理を全て `CommandSystem` 経由に切り替える。
    *   `TriggerSystem` が `PendingEffect` を処理する際、`EffectDef.commands` を参照して `CommandSystem` を呼び出すように変更する。
3.  **AI Input Update**:
    *   `TensorConverter` を更新し、`EffectActionType` ではなく、発生した `GameCommand` や盤面変化を観測するようにAI入力を刷新する。
4.  **GUI Update**:
    *   `CardEditor` を更新し、新スキーマ (`CommandDef`) の編集に対応させる（マクロ編集と詳細編集のハイブリッドUI）。
