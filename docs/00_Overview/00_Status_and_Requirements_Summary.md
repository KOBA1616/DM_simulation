# デュエル・マスターズ AI シミュレーター 要件定義書兼開発ステータス (2025-12-02版)

## 0. 本ドキュメントの運用方針 (Maintenance Policy)
本ドキュメントはプロジェクトのマスター要件定義書である。
**開発が完了した機能、仕様変更、または新たな要件が発生した場合は、必ず本ドキュメントを更新し、常に最新の状態を保つこと。**

## 1. プロジェクト概要
本プロジェクトは、TCG「デュエル・マスターズ」の高速かつ拡張性の高いシミュレーターを構築し、AlphaZero/MuZeroベースの高度なAIエージェントを育成することを目的とする。C++20による高速なコアエンジンと、Pythonによる柔軟な学習・GUI環境を統合する。

詳細なプロジェクト概要は [01_Project_Overview.md](./01_Project_Overview.md) を参照。

## 2. 現在の開発状況サマリ
- **フェーズ**: Phase 4 (Expansion) へ移行中。
- **ステータス**:
    - 学習パイプライン（データ収集→学習→検証）の動作検証完了。
    - ディレクトリ構成の整理完了 (不要ファイルの `archive/` への移動)。
    - 検証スクリプト (`verify_performance.py`) によるベンチマーク取得を確認済み。
- **直近の課題**:
    - **推論速度のボトルネック**: `lethal_puzzle_easy` 検証において、Pythonループ(MCTS 800)で180秒、C++ MCTS + Python推論で120秒/ゲームを要し、実用的な学習速度が出ていない。
    - **勝率0%**: 十分な探索回数(800)でもランダム初期化モデルではパズルを解けず、Draw（時間切れ）となる。
    - **レガシー資産**: `data/cards.csv` は `data/cards.json` に移行済みだが、一部のテスト (`tests/test_fuzzing_random_actions.py` 等) で依然として使用されている。
    - **対策**: Phase 5 (High-Performance Training Infrastructure) の実装を前倒しし、並列自己対戦 (Requirement 15) とC++推論パイプラインを確立する必要がある。

---

## 3. 完了した要件 (Completed Requirements)

以下の機能は実装済みであり、コードベースに存在することを確認済み。詳細は `docs/01_Completed_Specs/` 内の各ドキュメントを参照。

### 3.1 コアエンジン (C++20)
- [x] **高速ゲームエンジン**: ビットボード、メモリプール、ゼロコピー転送による高速化。
- [x] **基本ゲームルール**: マナチャージ、召喚、攻撃、ブロック、シールドブレイク、ターン進行。
- [x] **Result Stats基盤**: `CardStats` 構造体による統計集計。
- [x] **Scenario Mode基盤**: `ScenarioConfig` による盤面設定。
- [x] **汎用カードシステム (Generic Card System)**: JSON定義からのカード生成。

### 3.2 Python連携 & GUI
- [x] **Python Bindings (pybind11)**: `dm_ai_module` としてビルド済み。
- [x] **GUI (PyQt6)**: 盤面可視化機能（要PyQt6環境）。
- [x] **Basic AI**:
    - Python版 MCTS + MLP。
    - C++版 MCTS (`dm::ai::MCTS`) の実装とバインディング。
- [x] **Verification Script**: `verify_performance.py` をC++ MCTSを利用するように更新し、推論パイプラインを疎通させた。

---

## 4. 未開発・開発予定の要件 (Planned Requirements)

### 【優先度 1】基盤と堅牢性の確保 (Phase 1: Foundation & Robustness)
- [x] **環境安定化**: `import dm_ai_module` 安定化済み。
- [x] **ユニットテスト**: `pytest` による基本テスト実装済み。
- [x] **ディレクトリ整理**: 不要ファイルのアーカイブ化と整理完了。

### 【優先度 2】拡張性の確立 (Phase 2: Extensibility)
- [x] **JSON Loader / Card Editor**: 実装完了。

### 【優先度 3】小規模全体試験 (Phase 3: MVP Cycle)
- [x] **データ収集・学習・検証**: 完了。ただしパフォーマンスに課題あり。

### 【優先度 4】コンテンツ拡充と高度化 (Phase 4: Expansion)
- [ ] **カード量産**: 100種以上のカード実装。
- [ ] **高度AI機能**: PBT, League Training。

### 【優先度 5】ハイパフォーマンス学習基盤 (Phase 5: High-Performance Training Infrastructure)
**現在進行中 (In Progress)**

14. **C++ 特徴量抽出 & ゼロコピー (C++ Feature Extraction & Zero-Copy)**
    - [x] `TensorConverter::convert_to_tensor` 実装済み。
    - [x] `TensorConverter::convert_batch_flat` 実装済み。

15. **C++ Self-Play & Batch Inference**
    - [x] `dm::ai::MCTS` のC++実装完了。
    - [x] Pythonコールバックによるバッチ推論の疎通確認 (`verify_performance.py` で実証)。
    - [x] **ParallelRunner**: 複数スレッドで同時にMCTSを回し、GPUバッチサイズを埋める仕組みの実装完了 (`src/ai/self_play/parallel_runner.cpp`)。検証スクリプトによる動作確認中。

---

## 5. 実装ロードマップ (Implementation Order)

1.  **Optimize with C++ (Phase 5)**:
    - `ParallelRunner` の動作検証を行い、`verify_performance.py` または新規学習スクリプトに組み込んで学習速度を100倍以上（秒間数千手）にする。
2.  **Train Robust Model**:
    - 高速化した環境で `lethal_puzzle_easy` を解けるモデルを学習する。
3.  **Deploy to Cloud (Phase 6)**:
    - Kaggle等での大規模学習へ移行。
