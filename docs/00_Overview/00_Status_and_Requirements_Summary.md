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
AI領域では、探索ベースのリーサルソルバー（DFS）の実装が完了し、現在は進化型デッキ構築システム（Smart Evolution）と並列シミュレーション（ParallelRunner）の統合フェーズにあります。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **Card Owner Refactor**: `CardInstance` 構造体に `owner` フィールドを追加し、外部マップ `card_owner_map` を廃止しました。
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` を削除し、`GameLogicSystem` へ完全移行しました。
*   [Status: Done] **Action Generalization**: 全アクションハンドラーの `compile_action` 化が完了しました。
*   [Status: Done] **Dynamic Cost Reduction**: `ModifierDef` および `CostModifier` に `value_reference` フィールドを追加し、動的な値（`CARDS_DRAWN_THIS_TURN` 等）に基づくコスト計算を実現。
*   [Status: Done] **Revolution Change Logic**: 革命チェンジのロジックを実装し、攻撃時の入れ替え処理をサポートしました。
*   [Status: Done] **Hyper Energy**: ハイパー化（Hyper Energy）のコスト軽減ロジックおよびUIフラグを実装しました。

### 2.2 AI & ソルバー (`src/ai`)
*   [Status: Done] **Lethal Solver (DFS)**: `LethalDFS` クラスによる深さ優先探索ベースの詰み判定を実装。攻撃者とブロッカーのビットマスク管理による高速化を実現しています（トリガー等は現状未考慮）。
*   [Status: Done] **Parallel Runner**: OpenMPを用いた並列対戦実行環境 (`ParallelRunner`) を実装し、Pythonから制御可能にしました。

### 2.3 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Card Editor V2**: JSONベースのデータ駆動型エディタへの移行完了。
*   [Status: Done] **Neo/G-Neo UI**: Neoクリーチャー/G-Neoクリーチャーの選択および自動キーワード設定機能を追加。
*   [Status: Done] **Modifier Reference UI**: スタティックアビリティ編集画面にて「値参照（Value Reference）」の設定UIを追加。

### 2.4 学習基盤 (`dm_toolkit/training`)
*   [Status: WIP] **Smart Evolution Ecosystem**: デッキ進化システム (`evolution_ecosystem.py`) のプロトタイプを作成。「使用頻度」「リソース貢献度」に基づくスコアリングロジックを検証中。

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] AI Evolution & PBT (AI進化とPBT)
[Status: WIP]
単なる自己対戦だけでなく、Population Based Training (PBT) を用いた「メタゲーム進化」を実現します。

1.  [Status: Todo] **C++ Stats Integration**: `ParallelRunner` 内でカード使用統計（Win Contribution等）を直接集計し、Pythonへの転送コストを削減する。
2.  [Status: Todo] **PBT Pipeline**: 複数のエージェント（デッキ）を並列で対戦させ、勝者を交叉・変異させるPBTループを実装する。

### 3.2 [Priority: High] Search Solver Enhancement (探索ソルバーの高度化)
[Status: Todo]
現在の `LethalSolver` は純粋な盤面情報のみを使用していますが、不確定情報（シールドトリガー確率）を考慮したリスク評価を組み込みます。

1.  [Status: Todo] **Probabilistic Trigger Check**: 相手の公開デッキ情報からシールドトリガーの発動確率を計算し、期待値ベースで攻撃を判断するロジックを追加。

### 3.3 [Priority: Medium] GUI Advanced Features (GUI高度化)
[Status: Deferred]
エディタの機能拡充を行います。

1.  [Status: Todo] **Reaction Ability Editor**: `Ninja Strike` や `Strike Back` などの `ReactionAbility` を編集するための専用フォームを実装する。
2.  [Status: Todo] **Logic Mask System**: カードタイプや文明に基づき、矛盾する効果やアクションの組み合わせをUI上で禁止するバリデーション機能。

## 4. 今後の課題 (Future Tasks)

1.  [Status: Todo] **Transformer Architecture**: 現在の単純なResNet/MLPモデルから、自己注意機構（Self-Attention）を用いたTransformerモデルへの移行（Phase 4）。
2.  [Status: Todo] **G-Neo Robustness**: G-Neoの置換効果をハードコードから汎用イベントシステムへ移行。
3.  [Status: WIP] **Binding Restoration**: リファクタリングに伴い無効化された一部テストケースの復旧。
4.  [Status: Todo] **Phase 7 Implementation**: 新JSONスキーマへの完全移行（古いCSV/JSON互換性の廃止）。

## 5. 運用ルール (Operational Rules)
*   **テストコードの配置**: すべてのテストコード（Python）はプロジェクトルートの `tests/` ディレクトリに集約する。
*   **CI遵守**: `PyQt6` 依存テストはスキップし、必ずCIがグリーンになる状態でマージする。
*   **バインディング追従**: C++変更時は必ず `src/bindings/bindings.cpp` を更新する。
