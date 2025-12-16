
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
