# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## ステータス定義 (Status Definitions)
開発状況を追跡するため、各項目の先頭に以下のタグを付与してください。

*   `[Status: Todo]` : 未着手。AIが着手可能。
*   `[Status: WIP]` : (Work In Progress) 現在作業中。
*   `[Status: Review]` : 実装完了、人間のレビューまたはテスト待ち。
*   `[Status: Done]` : 完了・マージ済み。
*   `[Status: Blocked]` : 技術的課題や依存関係により停止中。
*   `[Status: Deferred]` : 次期フェーズへ延期
*   `[Test: Pending]` : テスト未作成。
*   `[Test: Pass]` : 検証済み。
*   `[Test: Fail]` : テスト失敗。修正が必要。
*   `[Test: Skip]` : 特定の理由でテストを除外中。
*   `[Known Issue]` : バグを認識した上でマージした項目。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

現在、**Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** がほぼ完了し、アクションハンドラーの統合と命令パイプラインの安定化（デバッグフェーズ）に移行しています。
また、並行して**Phase 3 (Self-Evolution)** に向けたAI学習パイプラインの整備と検証を進めています。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` クラスおよびファイルを物理的に削除しました。すべての呼び出し元 (`GameInstance`, `ActionGenerator`, `ScenarioExecutor`, `Bindings`) は `GameLogicSystem` へ移行されました。
*   [Status: Done] **GameLogicSystem Refactor**: アクションディスパッチロジックを `GameLogicSystem` に集約し、`PipelineExecutor` を介した処理フローを確立しました。
*   [Status: Done] **GameCommand**: 新エンジンの核となるコマンドシステム。`Transition`, `Mutate`, `Flow` に加え、`Stat` (統計更新), `GameResult` (勝敗判定), `Attach` (進化/クロス) を実装済み。
*   [Status: Done] **Instruction Pipeline**: `PipelineExecutor` が `GAME_ACTION` 命令 (`WIN_GAME`, `LOSE_GAME`, `TRIGGER_CHECK`, `STAT`更新) をサポートするように拡張されました。
*   [Status: Done] **Stack-Based VM**: `PipelineExecutor` を再帰ベースからスタックベースのVM（Virtual Machine）にリファクタリングし、`Instruction` 実行の一時停止と再開（Resume）を完全にサポートしました。
*   [Status: Done] **Complete Effect Resolution**: `EffectSystem::compile_effect` を実装し、複雑な効果（S・トリガー等）を `Instruction` リストにコンパイルして実行するフローを確立しました。
*   [Status: Done] **Action Generalization**: すべての `IActionHandler` に `compile_action` メソッドを追加し、インターフェースを統一しました。
*   [Status: Done] **Handler Migration**: 主要なハンドラー (`ManaHandler`, `DestroyHandler`, `SearchHandler` 等) は `compile_action` ベースに移行済み。
    *   *Update*: `MANA_CHARGE`は`MOVE_CARD`へ統合、`SEARCH_DECK`の変数リンク実装済み。
*   [Status: Done] **Gameplay Features**:
    *   **Revolution Change (革命チェンジ)**: 実装・検証済み (`test_revolution_change.py`)。
    *   **Hyper Energy (ハイパー化)**: 実装・検証済み (`test_hyper_energy.py`)。
    *   **Just Diver (ジャストダイバー)**: `PASSIVE_CONST` メカニズムで実装済み。
    *   **Meta Counter (メタカウンター)**: 手札誘発の実装済み (`test_meta_counter.py`)。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Status**: 稼働中 (Ver 2.3)。
*   [Status: Done] **Card Editor V2**: 3ペイン構成 (`CardDataManager`, `LogicTreeWidget`, `PropertyInspector`) への刷新完了。
*   [Status: Done] **Variable Linking**: アクション間で変数を渡す「Smart Link」UI実装済み。
*   [Status: Done] **Scenario Editor**: シナリオ作成・編集用GUI (`scenario_editor.py`) 実装済み。
*   [Status: Done] **Frontend Integration**: GUI (`app.py`) が `waiting_for_user_input` フラグを監視し、ゲームループを制御可能。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   [Status: WIP] **Status**: パイプライン稼働中。
*   [Status: Done] **Data Collection**: `DataCollector` (C++) と `collect_training_data.py` (Python) の連携完了。
*   [Status: Done] **Training Loop**: `train_simple.py` によるAlphaZero学習ループ実装済み。
*   [Status: Done] **Parallel Runner**: `ParallelRunner` によるマルチスレッドMCTS自己対戦の実装済み。
*   [Status: Todo] **Deck Evolution**: デッキ進化システム (`verify_deck_evolution.py`) はスタブ実装であり、本格的なロジック実装が必要。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: High] Phase 6: エンジン刷新 (Engine Overhaul) - 仕上げ
[Status: Review]
基盤移行は完了し、現在は安定化と統合テストのフェーズです。

*   **Step 4: Pure Command Generation (Current Focus)**
    *   [Status: Review] アクションハンドラーの `compile_action` 化は完了しました。
    *   [Status: Todo] **Execution Logic Verification**: 複雑なカード効果（連鎖する変数参照など）における命令パイプラインの挙動検証とバグ修正を継続します。

### 3.2 [Priority: Medium] Phase 7: データ完全移行 & クリーニング
[Status: Pending]
全てのデータを新形式へ移行し、レガシーコードを削除します。

*   **Step 1: データ構造の刷新 (Hybrid Schema)**
    *   [Status: Done] [Test: Pass]
    *   JSONスキーマに `CommandDef` を導入済み。
*   **Step 2: CommandSystem の実装**
    *   [Status: Done] [Test: Pass]

### 3.3 [Priority: Future] Phase 8: AI思考アーキテクチャの強化 (Advanced Thinking)
[Status: Deferred]
AIの「人間のような高度な思考」を実現するためのTransformer導入フェーズ。

*   **Transformer Architecture (NetworkV2)**: 未実装。
*   **Input Features**: 現在は固定長ベクトルを使用中。トークン系列ベースの特徴量抽出への移行が必要。

## 4. 今後の課題 (Future Tasks)

1.  [Status: Todo] **AI Model Enhancement**: Transformerアーキテクチャの実装と、トークンベースの特徴量抽出の実装 (Phase 8)。
2.  [Status: Todo] **Deck Evolution Ecosystem**: 単なる対戦だけでなく、メタゲームに合わせてデッキを自動調整する進化的アルゴリズムの実装 (Phase 3)。
3.  [Status: Todo] **Large Scale Verification**: 数万ゲーム規模の連続対戦によるエンジンの堅牢性検証と、メモリリークの完全な排除 (`ParallelRunner`の長時間稼働検証)。
4.  [Status: Review] **Logic Verification**: `SEARCH_DECK` 等の複雑なインタラクションが、命令パイプライン上で意図通り動作し続けるかの継続的な監視。

#### 新エンジン対応：Card Editor GUI構造の再定義 (Reference)

    (GUI実装はVer 2.3でほぼ完了していますが、将来的な拡張のために参照として保持します)
    *   **Ability Mode**: `TRIGGERED` vs `STATIC` の明確な区別。
    *   **Variable Linking**: `Input Source Key` UIによるアクションチェーンの可視化。

