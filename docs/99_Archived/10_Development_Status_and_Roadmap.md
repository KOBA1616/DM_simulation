# 10. 開発状況と今後のロードマップ (Development Status & Roadmap)

**注意: 本ドキュメントのロードマップは [20. 改定ロードマップ (Revised Roadmap)](./20_Revised_Roadmap.md) にて再定義されました。今後の開発は 20_Revised_Roadmap.md に従ってください。**

## 10.1 現在の開発段階 (Current Status: Phase 2.5)
**「基盤構築完了・高度化仕様策定フェーズ」**

現在、シミュレーターの核となるC++エンジンとPython GUIの連携は完了し、基本的な対戦が可能な状態にある。
直近の検討により、最強AI構築とカード量産のための**高度な機能群（AI/ML Specs, Card Generator）の要件定義が完了**したが、これらは**未実装**である。

### 実装済み機能 (Implemented)
- **Core Engine (C++20)**:
    - ビットボード、メモリプール、ゼロコピー転送による高速エンジン。
    - 基本的なゲームルール、フェーズ進行、ゾーン管理。
- **GUI (PyQt6)**:
    - 盤面表示、操作、MCTS可視化、ドック形式のレイアウト最適化。
    - `KeyboardInterrupt` 対応などのユーザビリティ向上。
- **Basic AI**:
    - 基本的なMCTS + MLP（AlphaZeroベース）の動作確認。

### 仕様策定済み・未実装機能 (Specified / Not Yet Implemented)
以下の機能は要件定義書（v2.2）に追加されたが、コード実装はこれから行う。
- **Advanced AI Architecture**:
    - **LLM Text Embeddings**: カードテキストのベクトル化入力。
- **Advanced Training Methods**:
    - **Auxiliary Tasks**: 勝敗以外の予測タスク。
    - **PBT**: ハイパーパラメータ自動最適化。
    - **PSRO**: メタゲームソルバー。
    - **League Training**: リーグ学習システム。
- **Meta-Game Evolution**:
    - **QD (MAP-Elites)**: 多様性探索。
    - **Co-evolution**: デッキとプレイングの共進化。
- **Development Tools**:
    - **Generic Card Generator**: データ駆動型カード実装システム（ECS/インタプリタ）。

## 10.2 今後の開発計画 (Future Roadmap)

*詳細は [docs/20_Revised_Roadmap.md](./20_Revised_Roadmap.md) を参照のこと。*

## 10.3 詳細実装計画 (Detailed Plan)
低級モデルやコンテキスト制限のある環境での開発を支援するため、タスクを極小単位に分解した詳細計画書を作成した。
今後の実装は原則として以下のドキュメントに従って進めること。

- **[11. 詳細実装計画書 (Detailed Implementation Steps)](./11_Detailed_Implementation_Steps.md)**

## 10.4 2025-12-01 現在の開発段階（要件定義書への記録）

この節は、2025-12-01 時点での実装進捗を要件定義書として残すためのまとめです。

- 実施済み（重要なマイルストーン）
    - `docs/` に PBT, POMDP（隠れ情報推論）, Meta-Game Curriculum, Result Stats, Scenario Training に関する設計仕様を追加。
    - C++ 側に `CardStats` 構造体を導入し、`GameState` に統計集計用のフィールドを追加（`global_card_stats`, `initial_deck_stats_sum`, `visible_stats_sum` 等）。
    - POMDP 補助関数 `on_card_reveal()`, `vectorize_card_stats()`, `get_library_potential()` を実装。
    - 初期化関数 `initialize_card_stats()` を実装し、カード DB 参照で `global_card_stats` エントリを確保する機能を追加。
    - 履歴統計の読み込み機能 `load_card_stats_from_json()` と、デッキから初期合算を計算する `compute_initial_deck_sums()` を追加（`data/card_stats.json` のサンプルあり）。
    - Python バインディングにラッパー実装を追加し、バインディング経由で初期化やベクトル化関数を呼べるようにした。
    - ローカルでの CMake ビルドを実行し、`dm_core` / `dm_sim_test` / `dm_ai_module` が生成されることを確認。

- 残課題 / 注意点
    - Windows 上での Python 拡張モジュールのロードにおいて、MinGW ビルドのランタイム DLL 依存で ImportError（DLL load failed）が発生。現在 PATH 調整等で診断中。
    - 実データ運用のために、履歴統計の収集・保存フォーマット（JSON/CSV）の最終仕様化が必要。

- 次の優先タスク（推奨順）
    1. Python の拡張モジュール読み込み問題を解決し、bindings の自動テストを動かす。
    2. 履歴統計フォーマットの確定と収集パイプラインの実装（トレーニング出力から統計を生成するスクリプト）。
    3. Python 側のユニットテスト (pytest) を追加して回帰を防止。

この記録は要件定義書の公式ログとして保存しました。

## 10.5 2025-12-05 開発進捗更新 (Development Update)

本節は、Phase 3 (Result Stats) および Phase 4 (Scenario Training) の初期実装完了に伴う記録です。

### 実施済み実装 (Completed Implementations)

#### 1. Result Stats System (Spec 15)
- **C++ Core**:
    - `CardStats` 構造体に `record_usage` メソッドを追加し、使用回数・早期/終盤使用率・トリガー発動率・コスト軽減額の集計ロジックを実装しました。
    - `GameState` クラスに `on_card_play` フックを追加し、カードプレイ時およびシールドトリガー使用時に統計を自動更新する仕組みを構築しました。
    - `EffectResolver` 内でカード使用イベントをフックし、統計システムへ通知するように変更しました。
- **Python Bindings**:
    - `get_card_stats` 関数を公開し、C++内で集計された統計データを Python 辞書形式で取得可能にしました。
- **Data Pipeline**:
    - `python/scripts/collect_stats.py` を作成しました。これにより、指定回数のランダム対戦（またはAI対戦）をバッチ実行し、集計結果を `data/card_stats_collected.json` に保存するパイプラインが確立されました。

#### 2. Scenario Training Mode (Spec 16)
- **C++ Core**:
    - `ScenarioConfig` 構造体を定義し、手札・マナ・バトルゾーン・シールド・墓地の状態を詳細に設定可能にしました。
    - `GameInstance` クラスを作成し、`reset_with_scenario(config)` メソッドを実装しました。これにより、任意の盤面状態（詰将棋や特定のコンボ始動盤面）からゲームを開始できるようになりました。
- **Python Bindings**:
    - `ScenarioConfig` および `GameInstance` を Python に公開し、スクリプトからシナリオを定義・実行可能にしました。
- **Testing**:
    - `tests/test_scenario.py` を作成し、シナリオモードが意図通りに盤面を構築できることを検証しました。

### 今後の課題 (Remaining Tasks)
1.  **Integration**: 強化学習ループ（Trainer）への統合は未実施です。今回作成した `GameInstance` を学習ワーカーに組み込む必要があります。

### 次のアクション (Next Actions)
- AI学習ループへのシナリオモードの統合（特定の失敗ケースをシナリオ化して反復練習させる）。

## 10.6 2025-12-06 開発進捗更新 (Development Update)

本節は、Phase 3およびPhase 4の残課題解消に伴う記録です。

### 実施済み実装 (Completed Implementations)

#### 1. Win Rate Logic (Spec 15)
- `GameState::on_game_finished` フックの実装を完了しました。「勝利貢献度 (Win Contribution)」、「逆転勝利 (Comeback Win)」、「決定打 (Finish Blow)」の計算ロジックが正常に機能することを確認しました。
- `PhaseManager::check_game_over` 内でゲーム終了を検知した際に、自動的に `on_game_finished` が呼び出されるように修正しました。また、`stats_recorded` フラグを導入し、統計の二重計上を防止しました。

#### 2. Scenario Shield Config (Spec 16)
- `ScenarioConfig` 構造体に `my_shields` フィールドを追加（確認）し、プレイヤー自身のシールド構成をカスタマイズ可能にしました。
- Python バインディングおよび `GameInstance` の初期化ロジックにおいても `my_shields` が正しく反映されることを検証しました（`tests/verify_shield_config.py` による検証済み）。

### 次のアクション (Next Actions)
- **Integration**: 強化学習トレーナー（RL Loop）において `GameInstance` と `ScenarioConfig` を活用し、特定の局面からの学習を開始する。

## 10.7 2025-12-07 ロードマップの再定義 (Roadmap Redefined)
ユーザー要件に基づき、堅牢性と拡張性を重視した新たなロードマップ `docs/20_Revised_Roadmap.md` を策定した。
今後の開発はこの新ロードマップに従い、フェーズ1（堅牢性確保）から順次進行する。

## 10.8 2025-12-07 開発進捗更新 (Development Update)

本節は、Revised Roadmap Phase 1 (Foundation & Robustness) の進捗記録です。

### 実施済み実装 (Completed Implementations)

#### 1. Engine & Unit Test Expansion (Phase 1 Task 1.2)
- **Unit Testing**:
    - `tests/test_engine_basics.py` を新規作成し、以下のコア機能が正常に動作することを検証しました。
        - **Mana Charge**: 手札からマナゾーンへのチャージ。
        - **Summon Creature**: マナコストを支払ってのクリーチャー召喚、`CardStats` のプレイ回数カウント更新。
        - **Attack & Block**: クリーチャーによるプレイヤーへの攻撃、シールドブレイク。
    - 特に、攻撃フェーズへの遷移とブロックフェーズ（パス時のバトル解決）のフローが正しく機能することを確認しました。

- **Engine Logic Fixes**:
    - `EffectResolver::resolve_block` において、ブロックアクション宣言後に `execute_battle` が呼び出されないバグを修正しました。これにより、ブロックフェーズでパス（ブロックなし）を選択した際に、正しく攻撃処理（シールドブレイクまたはクリーチャー破壊）が実行されるようになりました。
    - `ActionGenerator` 内のコメントを整理し、フェーズ遷移（MAIN -> ATTACK -> BLOCK）のロジックを明確化しました。

### 次のアクション (Next Actions)
- **Phase 1 Completion**:
    - 残る `GameState` ロジック（呪文の使用、トリガー処理等）のテストケース拡充。
    - CI/CDパイプラインでの自動テスト実行の安定化。
- **Phase 2 Preparation**:
    - JSONベースのカード定義システム（`GenericCardSystem`）の実装準備。
