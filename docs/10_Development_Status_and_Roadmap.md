# 10. 開発状況と今後のロードマップ (Development Status & Roadmap)

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

### Phase 3: AIコアの進化 (AI Core Evolution)
**目的**: 「結果スタッツ」と「非公開領域推論」を実装し、AIの基礎知能を飛躍的に向上させる。
1.  **Result Stats System (Spec 15)**:
    - C++エンジンに `CardStats` 構造体を実装し、16次元のスタッツを収集・ベクトル化する。
    - 人間のタグ付けを廃止し、データ駆動でのカード評価基盤を確立する。
2.  **POMDP & Distillation (Spec 13)**:
    - 相手の手札・シールドを推論するための Teacher-Student 蒸留システムを構築する。
    - C++エンジンに「自分自身の山札確率」を計算する `Self-Inference` ロジックを実装する。

### Phase 4: 高度な学習手法 (Advanced Training)
**目的**: 複雑なコンボやメタゲームの駆け引きを習得させる。
1.  **Scenario Training (Spec 16)**:
    - 特定盤面から開始する「詰将棋モード」を実装し、無限ループやリーサル手順を特訓する。
2.  **Meta-Game Curriculum (Spec 14)**:
    - アグロ/コントロールを交互に学習する「Dual Curriculum」を導入。
    - 苦手な相手と優先的に戦う「Adaptive League」を構築する。

### Phase 5: 自律進化エコシステム (Autonomous Ecosystem)
**目的**: 人間の介入なしに最強デッキを発見し続けるシステムを完成させる。
1.  **PBT & Kaggle Integration (Spec 12)**:
    - Kaggle Notebooks 上で動作する PBT (Population Based Training) 環境を構築。
    - 24時間稼働によるハイパーパラメータ探索とデッキ進化を実現する。

### Parallel Track: コンテンツ拡充 (Content Expansion)
**目的**: カードプールを拡大し、環境の多様性を確保する。
1.  **Generic Card Generator (Spec 9)**:
    - GUI操作とLLM補助によるカード量産ツールを開発する。
    - 既存カードをJSON定義へ完全移行する。

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
1.  **Win Rate Logic**: 現在 `on_game_finished` フックは空実装であり、勝敗統計（Win Contribution）はまだ正しく計算されません。これには「実際に勝利に貢献したカード」の追跡ロジックが必要です。
2.  **Scenario Shield Config**: 現在の `ScenarioConfig` では「自分のシールド」を設定するフィールドが不足しており、防御的なシナリオ（シールド0枚からの耐久など）の表現力が制限されています。
3.  **Integration**: 強化学習ループ（Trainer）への統合は未実施です。今回作成した `GameInstance` を学習ワーカーに組み込む必要があります。

### 次のアクション (Next Actions)
- `on_game_finished` の実装による勝率統計の完成。
- AI学習ループへのシナリオモードの統合（特定の失敗ケースをシナリオ化して反復練習させる）。
