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

現在、**Phase 1: Game Engine Reliability** および **Phase 6: Engine Overhaul** の実装が完了し、C++コア（`dm_ai_module`）のビルドは安定しています。
AI領域では、並列対戦環境の構築が完了し、次は不完全情報ゲームへの対応（Phase 2）とデッキ自動進化システム（Phase 3）の統合フェーズに移行します。

GUIエディタに関しては、ユーザー体験の向上と内部構造の刷新（ActionからCommandへの移行）を進めています。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **Card Owner Refactor**: `CardInstance` 構造体に `owner` フィールドを追加し、コントローラー判定をO(1)化。
*   [Status: Done] **EffectResolver Removal**: モノリシックな `EffectResolver` を削除し、処理を `IActionHandler` と `GameLogicSystem` へ委譲完了。
*   [Status: Done] **Action Generalization**: `GenericCardSystem` によるアクション処理の統一化完了。
*   [Status: Done] **Revolution Change**: 革命チェンジ（`ON_ATTACK_FROM_HAND` トリガーおよび入替処理）の実装完了。
*   [Status: Done] **Hyper Energy**: ハイパー化（コスト軽減およびクリーチャータップ）の実装完了。
*   [Status: Done] **Just Diver**: 「ジャストダイバー」などのターン経過依存の耐性ロジック実装完了。
*   [Status: Done] **Condition System**: `ConditionDef` および `ConditionSystem` による汎用的な発動条件判定の実装完了。

### 2.2 AI & ソルバー (`src/ai`)
*   [Status: Done] **Lethal Solver (DFS)**: 基本的な詰み判定（`LethalSolver`）の実装完了。ただし現在はヒューリスティックベースであり、厳密解ではない。
*   [Status: Done] **Parallel Runner**: OpenMPを用いた並列対戦環境 (`ParallelRunner`) の実装完了。`verify_performance.py` によるバッチ推論との連携を確認済み。
*   [Status: Done] **MCTS Implementation**: AlphaZero準拠のモンテカルロ木探索実装完了。
*   [Status: Review] **Inference System**: 相手の手札やシールド推論を行う `PimcGenerator` および `DeckInference` の基本実装と、Pythonバインディング・テストが完了（Phase 2）。

### 2.3 カードエディタ & ツール (`dm_toolkit/gui` / `python/gui`)
*   [Status: Done] **Card Editor V2**: データ構造をJSONツリーベースに刷新。
*   [Status: Done] **Variable Linking**: アクション間で変数を共有する「変数リンク機能」の実装完了。
*   [Status: Done] **Condition Editor**: GUI上で発動条件（Condition）を編集するフォームの実装完了。
*   [Status: Done] **Simulation GUI**: GUIから並列シミュレーションを実行するダイアログの実装完了。
*   [Status: Done] **UI Terminology Unification (Action/Command)**: GUI上の表示名を「アクション (Action)」に統一し、内部構造（Command/Legacy Action）の違いをユーザーから隠蔽完了。選択肢リストも統合済み。

### 2.4 学習基盤 (`python/training`)
*   [Status: Done] **Training Loop**: `collect_training_data.py` -> `train_simple.py` の基本ループ構築完了。
*   [Status: Done] **Scenario Runner**: `ScenarioExecutor` を用いた定石シミュレーション機能の実装完了。

## 3. UI/UX方針と内部構造の乖離 (UI/UX Strategy & Internal Divergence)

カードエディタの発展に伴い、内部実装とユーザーインターフェースの間で以下の戦略を採用します。

### 3.1 アクションとコマンドの融合 (Action/Command Fusion)
*   **GUI表示名**: **「アクション (Action)」** に統一します。
    *   理由: カードゲームの文脈において「コマンド」より「アクション」の方が自然であり、ユーザーの認知負荷を下げるため。
*   **内部構造**: **「コマンド (Command)」** (C++ `GameCommand`) への移行を推進します。
    *   理由: `ActionDef` (旧) よりも拡張性（分岐、フロー制御）が高いため。
    *   現状は `Legacy Action` と `New Command` が混在する過渡期ですが、GUI上では統合されたリスト（`UNIFIED_ACTION_TYPES`）を提供します。

### 3.2 AI開発上の注意点 (Note for AI Development)
*   **用語の乖離**: GUI上の「アクション」は、コード上の `ActionDef` (旧) と `CommandDef` (新) の両方を指す包括的な用語として扱います。
*   **プロンプト指示**: AIに対してコード生成や修正を指示する際は、「GUI上のアクションは、内部的にはCommandパターン（またはLegacy Action）で実装されている」というマッピングルールを前提としてください。

## 4. ロードマップ (Roadmap)

### 4.1 [Priority: Critical] AI Evolution System (AI進化システム)
[Status: Review]
Phase 3の中核となる「自筆進化エコシステム」を構築します。

1.  [Status: Done] **Deck Evolution (PBT)**: `verify_deck_evolution.py` (現在は `evolve_meta.py`) をPythonベースのPBTシステムとして再実装完了。`MockParallelRunner` による動作検証済み。
2.  [Status: Done] **Meta-Game Integration**: 学習したモデルとデッキ情報を `meta_decks.json` に自動反映し、次世代の学習にフィードバックするループを構築完了。

### 4.2 [Priority: High] Model Architecture (モデルアーキテクチャ)
[Status: Todo]
Phase 4の要件である高性能モデルへの移行を行います。

1.  [Status: Todo] **Transformer Implementation**: 現在のResNet/MLPモデルから、Self-Attentionを用いたTransformerモデルへ移行し、盤面の文脈理解能力を向上させる。

### 4.3 [Priority: Medium] Engine Robustness (エンジン堅牢化)
[Status: Todo]
エッジケースへの対応と安定性向上を図ります。

1.  [Status: WIP] **Action→Command 完全移行 (Legacy Action Migration)**: GUIの「表示はアクション、内部はコマンド」を維持したまま、保存形式・テスト・周辺ツールを `commands` 正本へ統一し、`actions` は互換入力としてのみ扱う。
    *   [Status: Done] **Load-Lift (読み込み時変換)**: 読み込み時に `effects[].actions` を `effects[].commands` に変換し、エディタ内部では `actions` を保持しない。
    *   [Status: Done] **Save Commands-only (保存時コマンドのみ出力)**: 保存・再構築時に `commands` のみを書き出し、`actions` は出力しない（過渡期の互換出力は「エクスポート専用」に限定する）。
    *   [Status: WIP] **GUI 表示・プレビューの commands-first 化**: プレビュー/自然文生成が `actions` 参照で落ちないよう、`commands` を一次ソースとして扱い、`actions` は fallback のみにする。
    *   [Status: Todo] **テスト整流 (commands-only 方針への追従)**: `actions` を期待しているGUI/統合テストを `commands` 期待へ移行し、方針とテストの整合を取る。
    *   [Status: Todo] **CI ガード**: `data/` 配下のカードJSONに `actions` フィールドが残っていないことを検査し、差し戻しを防ぐ。
    *   [Status: Todo] **未対応アクションの棚卸しと優先度付け**: 未対応 `Action type` をノイズ無しで集計し、(1) Engine生成系 → (2) 頻出カード効果 → (3) 低頻度/別名 の順で変換対応を進める。
    *   [Status: Todo] **実行経路の一本化**: AI/GUI/スクリプトのどの入口からも「実行できるCommand」として扱えるよう、`wrap_action`/ネイティブコマンド/辞書表現の責務を整理する。

    **受け入れ基準 (Acceptance Criteria)**
    *   `data/` 配下のカードJSONに `actions` キーが存在しない（互換用途の `editor_templates` などは例外扱いを明記）。
    *   GUIエディタでカードを開いた場合、Effectは `commands` のみを持ち、変換不能は `legacy_warning` / Warning 表示で可視化される。
    *   保存時に `commands` のみが出力され、スキーマ最低条件（type/必須フィールド/枝・optionsの形）が満たされる。
    *   プレビュー/カードテキスト生成は `commands` だけでも破綻しない（`actions` が無いカードでも表示できる）。
	
    **テスト方針 (Test Policy)**
    *   [Test: Pending] Load-Lift の単体テストに加え、GUI保存I/Oが `commands` を出力する統合テストを追加する。
    *   [Test: Pending] `data/` スキーマ検証（必須キー、ネスト構造）と `actions` 排除チェックをCIで常時実行する。
2.  [Status: Todo] **Strict Lethal Solver**: 現在のヒューリスティック版から、ルールを完全に厳密にシミュレートするソルバーへ移行する。
3.  [Status: Todo] **Memory Leak Fix**: `ParallelRunner` を長時間/大量スレッドで実行した際に発生する `std::bad_alloc` (メモリリーク) の調査と修正。

### 4.4 [Priority: Low] GUI Expansion (GUI拡張)
[Status: Deferred]

1.  [Status: Todo] **Reaction Ability Editor**: ニンジャ・ストライク等の `ReactionAbility` 編集UIの実装。
2.  [Status: Todo] **Logic Mask**: 矛盾する効果の組み合わせを防止するバリデーション機能。

## 5. 開発スケジュール・今後の予定 (Development Schedule)

1.  **直近の目標 (Phase 2-3 Integration)**:
    *   不完全情報推論（Inference System）の実装開始。
    *   デッキ進化システム（PBT）のパイプライン化。
2.  **中期的目標 (Phase 4)**:
    *   Transformerモデルの実装と学習。
3.  **長期的目標 (Phase 5)**:
    *   RNN/LSTMを用いた時系列データの活用と、PPOなどの強化学習アルゴリズムの導入。

## 6. 既知の問題 (Known Issues)

*   [Known Issue] **ParallelRunner Memory Leak**: 大規模な並列シミュレーション時にメモリリークが発生する可能性がある。
*   [Known Issue] **Inference Heuristic**: 現在の推論ロジックは単純な確率に基づいており、ブラフやメタゲームを考慮していない。

## 7. 運用ルール (Operational Rules)
*   **コミットメッセージ**: 日本語で記述する。
*   **ソースコード**: コメントも含めUTF-8で統一する。
*   **テスト**: `python/tests/` 以下のpytestを実行し、CIを通過させること。
