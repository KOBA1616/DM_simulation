# デュエル・マスターズ AI シミュレーター 要件定義書兼開発ステータス (2025-12-02版)

## 1. プロジェクト概要
本プロジェクトは、TCG「デュエル・マスターズ」の高速かつ拡張性の高いシミュレーターを構築し、AlphaZero/MuZeroベースの高度なAIエージェントを育成することを目的とする。C++20による高速なコアエンジンと、Pythonによる柔軟な学習・GUI環境を統合する。

詳細なプロジェクト概要は [01_Project_Overview.md](./01_Project_Overview.md) を参照。

## 2. 現在の開発状況サマリ
- **フェーズ**: Phase 2.5（基盤構築完了・高度化仕様策定）から Phase 3（MVPサイクル）への移行期。
- **ステータス**: コアエンジンとGUIの連携は完了し、対戦可能。高度な学習機能（Result Stats, Scenario Mode）の初期実装が完了。
- **直近の課題**: Windows環境におけるPython拡張モジュールのDLL読み込みエラーの解消、およびカード追加プロセスの効率化（JSON化）。

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
    - [ ] **Python Import Issueの解決**: Windows/MinGW環境でのDLLロードエラーを解消し、`import dm_ai_module` を安定させる。
    - [ ] **CI/CDパイプライン**: ビルドとテストの自動化を安定させる。
2.  **ユニットテスト拡充 (Unit Test Expansion)**
    - [ ] **Core Logic Tests**: マナチャージ、シールドトリガー等の基本動作を検証する `pytest` の追加。
    - [ ] **CardStats Verification**: 実装された統計機能が正しく数値を集計しているかのテスト。

### 【優先度 2】拡張性の確立 (Phase 2: Extensibility)
**目的**: エンジンコードを修正せずに、GUIからカードを追加できる「データ駆動型」環境を構築する。

3.  **汎用カードシステム (Generic Card System)**
    - [ ] **JSON Loader**: `src/core/card_json_types.hpp` に基づき、JSON定義からカード効果を生成するC++ロジックの実装。
    - 参照: [09_Card_Generator_Architecture.md](../02_Planned_Specs/09_Card_Generator_Architecture.md)
4.  **GUIカードエディタ (Card Editor)**
    - [ ] **JSON Editor**: 既存のCSVベースのエディタを刷新し、JSON形式で効果（Trigger, Effect）を編集できるGUIツールの開発。
    - [ ] **Integration**: エディタで作成したカードを即座にエンジンにロードしてテストする機能。

### 【優先度 3】小規模全体試験 (Phase 3: MVP Cycle)
**目的**: シナリオモードを用いて、AIが「学習によって強くなる」サイクルを実証する。

5.  **データ収集パイプライン (Data Collection)**
    - [ ] **Scenario Loop**: 特定のシナリオ（詰めろ盤面など）を繰り返しプレイし、統計データと勝敗を収集するスクリプト。
6.  **簡易学習ループ (Simple Training)**
    - [ ] **Training Script**: 収集したデータを用いてモデルを更新する学習ループの実装。
7.  **性能検証 (Verification)**
    - [ ] **Impact Analysis**: 学習前後でのシナリオクリア率（勝率）の向上を定量的に確認する。

### 【優先度 4】コンテンツ拡充と高度化 (Phase 4: Expansion)
**目的**: 量産と高度なAI技術の導入。

8.  **カード量産 (Mass Production)**
    - [ ] 主要カード（100種〜）のJSON化と実装。
9.  **高度AI機能 (Advanced AI)**
    - [ ] **PBT (Population Based Training)**: ハイパーパラメータ自動最適化の実装。
        - 参照: [12_PBT_Design_Spec.md](../02_Planned_Specs/12_PBT_Design_Spec.md)
    - [ ] **League Training**: 過去の自分との対戦（リーグ戦）の実装。
        - 参照: [14_Meta_Game_Curriculum_Spec.md](../02_Planned_Specs/14_Meta_Game_Curriculum_Spec.md)
    - [ ] **Full POMDP**: 推論結果をAIの入力特徴量として完全統合。
        - 参照: [13_POMDP_Inference_Spec.md](../02_Planned_Specs/13_POMDP_Inference_Spec.md)

---

## 5. 実装ロードマップ (Implementation Order)

上記の優先順位に基づき、以下の順序で着手することを推奨します。

1.  **Fix DLL Error**: まずPythonからC++モジュールを呼べない問題を解決する（最優先）。
2.  **Test CardStats**: 統計機能が正しいことをテストで保証する。
3.  **Implement JSON Loader**: C++側でJSONを読めるようにする。
4.  **Update Card Editor**: GUIでJSONを作れるようにする。
5.  **Run Scenario Training**: シナリオモードで学習サイクルを回してみる。

詳細な実装手順は [17_Detailed_Implementation_Instructions_for_AI.md](../02_Planned_Specs/17_Detailed_Implementation_Instructions_for_AI.md) を参照してください。
