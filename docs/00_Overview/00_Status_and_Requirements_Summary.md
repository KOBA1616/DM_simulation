# デュエル・マスターズ AI シミュレーター 要件定義書兼開発ステータス (2025-12-02版)

## 0. 本ドキュメントの運用方針 (Maintenance Policy)
本ドキュメントはプロジェクトのマスター要件定義書である。
**開発が完了した機能、仕様変更、または新たな要件が発生した場合は、必ず本ドキュメントを更新し、常に最新の状態を保つこと。**

## 1. プロジェクト概要
本プロジェクトは、TCG「デュエル・マスターズ」の高速かつ拡張性の高いシミュレーターを構築し、AlphaZero/MuZeroベースの高度なAIエージェントを育成することを目的とする。C++20による高速なコアエンジンと、Pythonによる柔軟な学習・GUI環境を統合する。

詳細なプロジェクト概要は [01_Project_Overview.md](./01_Project_Overview.md) を参照。

## 2. 現在の開発状況サマリ
- **フェーズ**: Phase 3 (MVP Cycle) 開始。
- **ステータス**: Phase 3 (MVP Cycle) 完了、Phase 4 (Expansion) へ移行準備中。
- **直近の課題**: データ収集と学習の反復による、有意な勝率向上(Clear Rate > 0%)の実証。

今後のロードマップ詳細は [20_Revised_Roadmap.md](./20_Revised_Roadmap.md) を参照。

---

## 3. 完了した要件 (Completed Requirements)

以下の機能は実装済みであり、コードベースに存在することを確認済み。詳細は `docs/01_Completed_Specs/` 内の各ドキュメントを参照。

### 3.1 コアエンジン (C++20)
- [x] **高速ゲームエンジン**: ビットボード、メモリプール、ゼロコピー転送による高速化。
    - 参照: [03_Core_Data_Specs.md](../01_Completed_Specs/03_Core_Data_Specs.md)
- [x] **基本ゲームルール**: マナチャージ、召喚、攻撃、ブロック、シールドブレイク、ターン進行。
    - 参照: [04_Game_Rules.md](../01_Completed_Specs/04_Game_Rules.md)
- [x] **Result Stats基盤**:
    - `CardStats` 構造体による16次元スタッツ（使用率、勝率貢献度など）の定義。
    - `GameState` への統計集計フック（`on_card_play` 等）の実装。
    - 参照: [15_Result_Stats_Spec.md](../01_Completed_Specs/15_Result_Stats_Spec.md)
- [x] **Scenario Mode基盤**:
    - `ScenarioConfig` 構造体による盤面状態（手札、マナ、シールド等）の定義。
    - `GameInstance::reset_with_scenario` による任意盤面からのゲーム開始機能。
    - 参照: [16_Scenario_Training_Spec.md](../01_Completed_Specs/16_Scenario_Training_Spec.md)
- [x] **汎用カードシステム (Generic Card System)**:
    - **JSON Loader**: `src/core/card_json_types.hpp` に基づき、JSON定義からカード効果を生成するC++ロジックの実装。
    - 参照: [09_Card_Generator_Architecture.md](../02_Planned_Specs/09_Card_Generator_Architecture.md)

### 3.2 Python連携 & GUI
- [x] **Python Bindings (pybind11)**: C++エンジンをPythonモジュールとしてビルド・公開。
    - 参照: [08_Cpp_Integration.md](../01_Completed_Specs/08_Cpp_Integration.md)
- [x] **GUI (PyQt6)**: 盤面可視化、操作、MCTS探索木の可視化。
    - 参照: [07_Frontend_DevOps.md](../01_Completed_Specs/07_Frontend_DevOps.md)
- [x] **Basic AI**: MCTS + MLP（AlphaZeroベース）の推論・実行。
- [x] **POMDP (部分観測マルコフ決定過程) 初期実装**:
    - `ParametricBelief` クラスによる非公開領域（手札・デッキ）の確率的推論ロジック。

---

## 4. 未開発・開発予定の要件 (Planned Requirements)

未実装の要件を、**「堅牢性」「拡張性」「実証」「高度化」**の4段階の優先順位で整理しました。詳細は `docs/02_Planned_Specs/` 内の各ドキュメントを参照。

### 【優先度 1】基盤と堅牢性の確保 (Phase 1: Foundation & Robustness)
**目的**: 開発環境の不具合を解消し、エンジンの動作を保証するテスト基盤を固める。

1.  **環境安定化 (Environment Stability)**
    - [x] **Python Import Issueの解決**: Windows/MinGW環境でのDLLロードエラーを解消し、`import dm_ai_module` を安定させる。（Linux環境では動作確認済み）
    - [ ] **CI/CDパイプライン**: ビルドとテストの自動化を安定させる。
2.  **ユニットテスト拡充 (Unit Test Expansion)**
    - [x] **Core Logic Tests**: マナチャージ、シールドトリガー等の基本動作を検証する `pytest` の追加。
    - [x] **CardStats Verification**: 実装された統計機能が正しく数値を集計しているかのテスト。
3.  **ストレステスト (Fuzzing / Stress Test)**
    - [ ] **Random Action Fuzzing**: `ActionGenerator` が生成する合法手の中から、完全にランダム（あるいは意地悪）な操作を高速で何万回も行い、assertエラーやクラッシュが発生しないか監視するスクリプトを作成する。

### 【優先度 2】拡張性の確立 (Phase 2: Extensibility)
**目的**: エンジンコードを修正せずに、GUIからカードを追加できる「データ駆動型」環境を構築する。

4.  **汎用カードシステム (Generic Card System)** (Completed)
    - [x] **JSON Loader**: 実装完了。`tests/test_json_loader.py` で動作検証済み。
5.  **GUIカードエディタ (Card Editor)** (Completed)
    - [x] **JSON Editor**: 既存のCSVベースのエディタを刷新し、JSON形式で効果（Trigger, Effect）を編集できるGUIツールの開発。
    - [x] **Integration**: エディタで作成したカードを即座にエンジンにロードしてテストする機能。
6.  **デバッグ機能強化 (Enhanced Debugging)**
    - [ ] **Internal State Debugging**: C++側でも非公開領域（相手の手札やシールドの中身など）のオブジェクト情報にアクセスできるデバッグ機能を強化する。
    - [ ] **Event Hooks**: ゲーム開始時、終了時、特定アクション時などのイベントフックを拡充し、詳細なログ出力や状態監視を可能にする。

### 【優先度 3】小規模全体試験 (Phase 3: MVP Cycle)
**目的**: シナリオモードを用いて、AIが「学習によって強くなる」サイクルを実証する。

7.  **データ収集パイプライン (Data Collection)** (Done)
    - [x] **Scenario Loop**: 特定のシナリオ（詰めろ盤面など）を繰り返しプレイし、統計データと勝敗を収集するスクリプト。
8.  **簡易学習ループ (Simple Training)** (Done)
    - [x] **Training Script**: 収集したデータを用いてモデルを更新する学習ループの実装。
        - `python/training/collect_training_data.py`: MCTS自己対戦によるデータ収集。
        - `python/training/train_simple.py`: 収集データを用いたAlphaZeroモデル(MLP)の学習。
9.  **性能検証 (Verification)** (Done)
    - [x] **Verification Script**: `python/training/verify_performance.py` 実装済み。
    - [x] **Impact Analysis**: 学習前後でのシナリオクリア率（勝率）の向上を定量的に確認する基盤が完了。
10. **シナリオモード改善 (Scenario Mode Improvement)**
    - [ ] **GUI Scenario Setup**: GUI上で特定のシチュエーション（盤面、手札、シールド等）を自由に設定・保存できる機能。
    - [ ] **Randomized Scenario Training**: 指定されたデッキからランダムにカードを展開して初期盤面を生成し、多様な状況で学習できる仕組み。
    - [ ] **Hidden Information Handling**: C++側は正解のシールドトリガー情報を持つが、AI側にはそれを「未知のカード」として扱い、確率的な推論（POMDP）を行わせる仕組みの実装。
    - [ ] **Strategy Improvement**: シールドトリガーの軽視や、相手の公開領域からのカウンター予測（リーサル判断）の弱さを克服するための報酬設計や特徴量追加。

### 【優先度 4】コンテンツ拡充と高度化 (Phase 4: Expansion)
**目的**: 量産と高度なAI技術の導入。

11. **カード量産 (Mass Production)**
    - [ ] 主要カード（100種〜）のJSON化と実装。
12. **ゲームルール制約 (Game Rule Constraints)**
    - [ ] **Deck Construction Rules**: 同名カード4枚制限などのデッキ構築ルールをエンジンレベルで強制・検証する機能。
13. **高度AI機能 (Advanced AI)**
    - [ ] **PBT (Population Based Training)**: ハイパーパラメータ自動最適化の実装。
        - 参照: [12_PBT_Design_Spec.md](../02_Planned_Specs/12_PBT_Design_Spec.md)
    - [ ] **League Training**: 過去の自分との対戦（リーグ戦）の実装。
        - 参照: [14_Meta_Game_Curriculum_Spec.md](../02_Planned_Specs/14_Meta_Game_Curriculum_Spec.md)
    - [ ] **Full POMDP**: 推論結果をAIの入力特徴量として完全統合。
        - 参照: [13_POMDP_Inference_Spec.md](../02_Planned_Specs/13_POMDP_Inference_Spec.md)

### 【優先度 5】ハイパフォーマンス学習基盤 (Phase 5: High-Performance Training Infrastructure)
**目的**: Pythonの柔軟性とC++の速度を融合し、GPU使用率を最大化する「最強の学習環境」を構築する。

14. **C++ 特徴量抽出 & ゼロコピー (C++ Feature Extraction & Zero-Copy)**
    - **目的**: 盤面状態(`GameState`)からニューラルネットワーク入力(`Tensor`)への変換をC++で行い、Python側の負荷とデータ転送コストを削減する。
    - **内容**:
        - `GameState::to_feature_vector()` を実装。
        - **Zero-Copy Tensor**: Pythonには `numpy` 配列（ゼロコピー）としてポインタ渡しを行い、メモリコピーのオーバーヘッドを排除する。

15. **C++ Self-Play & Batch Inference (C++ Self-Play & Batch Inference)**
    - **目的**: Pythonで1手ずつループするのではなく、C++側で対戦を完結させ、推論時のみPythonにバッチリクエストを送る構造にする。
    - **アーキテクチャ**:
        1.  **C++ Workers**: 複数の対戦（MCTS）がC++スレッド内で並列進行。
        2.  **Inference Queue**: 推論が必要な局面データをキューに積む。
        3.  **Python Batch Server**: Python側でキューを監視し、まとめてGPU推論を行い、結果をC++側に返す。
    - **メリット**: PythonのGIL（Global Interpreter Lock）の影響を最小限に抑え、GPU使用率を最大化する。

16. **ベクトル化環境API (Vectorized Environment API)**
    - **目的**: 上記MCTSを管理し、N個の並列対戦を一括で進めるインターフェース。
    - **内容**: `VectorGameInstance` クラス。`step_async()` で全環境のMCTSを開始し、`await_results()` で全環境のアクション決定を待つ等の制御を行う。

17. **C++ リプレイバッファ (C++ Replay Buffer)**
    - **目的**: 生成された大量の学習データ（数百万ステップ分）をPythonのリストではなく、C++のメモリプールで管理し、GCオーバーヘッドを回避する。
    - **内容**: `add_trajectory()` で対戦ログを格納し、`sample_batch()` で学習用ミニバッチを高速に生成してPythonに渡す。

14. **追加要件 (Optimization Requirements)**
    - [ ] **バイナリログ形式 (Binary Replay Format)**
        - 1試合を数KBに圧縮するバイナリ形式の策定と、C++側での高速書き出し/Python側での高速読み込みの実装。
    - [ ] **確定動作の高速化 (Deterministic Fast-Forward)**
        - 合法手が1つしかない場合、AI推論をスキップして即時実行するオプションの実装。

### 【優先度 6】クラウド・エコシステム (Phase 6: Cloud Ecosystem)
**目的**: ローカルPCのリソース制約から解放され、Kaggle等のクラウドGPUリソースを活用して大規模学習を行う。

15. **Kaggle環境対応 (Kaggle Integration)**
    - **目的**: Kaggle Notebook上でC++エンジンと学習ループを動作させる。
    - **内容**:
        - **Setup Script**: Kaggle環境（Linux）でCMakeビルドとPython拡張インストールを自動で行うスクリプト。
        - **Dataset Sync**: ソースコードや最新モデルをKaggle Datasetとして自動アップロードするCIワークフロー。
        - **Output Management**: 学習済みモデルやログをWandB (Weights & Biases) 等の外部ストレージに転送する仕組み。

16. **クラウド自動化パイプライン (Cloud Automation)**
    - **目的**: ローカルからコマンド一つでクラウド上の学習ジョブを開始・回収する。
    - **内容**: Kaggle API を利用し、ローカルの更新をプッシュ→クラウドで学習→結果をプルするサイクルの自動化。

---

## 5. 実装ロードマップ (Implementation Order)

上記の優先順位に基づき、以下の順序で着手することを推奨します。

1.  **Fix DLL Error**: まずPythonからC++モジュールを呼べない問題を解決する（最優先）。(Linux環境では完了)
2.  **Test CardStats**: 統計機能が正しいことをテストで保証する。(完了)
3.  **Implement JSON Loader**: C++側でJSONを読めるようにする。
4.  **Update Card Editor**: GUIでJSONを作れるようにする。
5.  **Run Scenario Training**: シナリオモードで学習サイクルを回してみる。
6.  **Optimize with C++**: Phase 5の機能（特徴量抽出 -> C++ MCTS -> バッチ化）を順次実装し、学習速度を向上させる。
7.  **Deploy to Cloud**: Phase 6のKaggle連携を行い、大規模学習を開始する。

詳細な実装手順は [17_Detailed_Implementation_Instructions_for_AI.md](../02_Planned_Specs/17_Detailed_Implementation_Instructions_for_AI.md) を参照してください。
