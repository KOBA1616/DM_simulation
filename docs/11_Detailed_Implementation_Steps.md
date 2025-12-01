# 11. 詳細実装計画書 (Detailed Implementation Steps)

本ドキュメントは、要件定義書（Spec 12～16）に基づき、AIの高度化と自律進化システムを実装するための具体的なステップを定義する。

## Phase 3: AIコアの進化 (AI Core Evolution)

### Step 1: 結果スタッツ基盤の実装 (Result Stats)
*   **参照**: [15. 結果スタッツ設計書](./15_Result_Stats_Spec.md)
*   **対象ファイル**: `src/core/card_stats.hpp`, `src/engine/game_state.cpp`
*   **タスク**:
    1.  `CardStats` 構造体を定義し、16次元のスタッツ（Early Usage, Hand Adv等）を蓄積するメンバ変数を追加する。
    2.  `GameState` クラスに `global_card_stats` マップを追加し、対戦終了時に結果を集計するロジックを実装する。
    3.  `vectorize_card_stats(card_id)` 関数を実装し、正規化された16次元ベクトルを返すようにする。
    4.  **Convergence Strategy**: 学習初期（Mask Phase）のために、全スタッツを0で埋めるフラグまたは引数を `vectorize` 関数に追加する。

### Step 2: 非公開領域推論の実装 (POMDP / Self-Inference)
*   **参照**: [13. 非公開領域推論システム設計書](./13_POMDP_Inference_Spec.md)
*   **対象ファイル**: `src/engine/game_state.cpp`
*   **タスク**:
    1.  `GameState` に `initial_deck_distribution`（初期デッキのスタッツ合計）と `visible_card_stats`（公開領域のスタッツ合計）を保持させる。
    2.  カードが公開領域（手札、マナ、墓地、場）に移動するたびに `visible_card_stats` を差分更新する。
    3.  `get_library_potential()` 関数を実装し、`(Initial - Visible) / Remaining` で山札の期待値ベクトルを計算する。

### Step 3: 知識の蒸留システムの実装 (Teacher-Student Distillation)
*   **参照**: [13. 非公開領域推論システム設計書](./13_POMDP_Inference_Spec.md)
*   **対象ファイル**: `python/models/network.py`, `python/scripts/train.py`
*   **タスク**:
    1.  **Teacher Model**: 入力層に「相手の手札・シールド情報」を含めたモデル定義を作成する。
    2.  **Student Model**: 通常の入力のみを受け取るモデル定義を作成する。
    3.  学習ループ (`train_step`) に `KL_div` 損失関数を追加し、Teacherの出力分布（Logits）にStudentを近づける蒸留プロセスを実装する。

---

## Phase 4: 高度な学習手法 (Advanced Training)

### Step 1: シナリオモードの実装 (Scenario Mode)
*   **参照**: [16. シナリオ・トレーニング設計書](./16_Scenario_Training_Spec.md)
*   **対象ファイル**: `src/engine/game_instance.cpp`, `src/core/scenario_config.hpp`
*   **タスク**:
    1.  `ScenarioConfig` 構造体を定義（手札、マナ、盤面の指定）。
    2.  `reset_with_scenario(config)` 関数を実装し、指定された状態でゲームを開始できるようにする。
    3.  Pythonバインディング (`dm_ai_module`) に `reset_scenario` を公開する。
    4.  **Batch Optimization**: シナリオモードの並列実行を高速化するため、`GameInstance` プールを活用する仕組みを検討する。

### Step 2: ループ検知と特訓ループ (Loop Detection & Drill)
*   **参照**: [16. シナリオ・トレーニング設計書](./16_Scenario_Training_Spec.md)
*   **対象ファイル**: `src/engine/game_state.cpp`, `python/training/scenario_runner.py`
*   **タスク**:
    1.  **C++**: `calculate_hash()` を実装し、同一局面が3回続いたら `loop_proven` フラグを立てて勝利とする。
    2.  **Python**: `SCENARIOS` 辞書にコンボ練習用の盤面定義を記述する。
    3.  **Python**: シナリオ専用の学習ループを実装し、ループ証明成功時に高報酬を与える。

### Step 3: メタゲーム・カリキュラムの実装 (Curriculum & League)
*   **参照**: [14. メタゲーム・カリキュラム設計書](./14_Meta_Game_Curriculum_Spec.md)
*   **対象ファイル**: `python/training/curriculum.py`, `python/training/league_manager.py`
*   **タスク**:
    1.  **Dual Curriculum**: エピソード数に応じて「アグロデッキ/報酬」と「コントロールデッキ/報酬」を切り替えるロジックを実装。
    2.  **League Manager**: 過去のモデルファイル (`.pth`) をロードし、勝率の低い相手を優先的にサンプリングする `get_opponent()` を実装。

---

## Phase 5: 自律進化エコシステム (Autonomous Ecosystem)

### Step 1: PBTワーカーの実装 (Population Based Training)
*   **参照**: [12. PBT導入設計書](./12_PBT_Design_Spec.md)
*   **対象ファイル**: `python/training/pbt_worker.py`
*   **タスク**:
    1.  ハイパーパラメータ（学習率、割引率、エントロピー係数）をJSONからロードして学習を開始するワーカークラスを作成。
    2.  一定期間ごとに `eval_performance()` を実行し、結果を共有ストレージ（CSV等）に書き込む。
    3.  **Exploit & Explore**: 下位モデルの場合、上位モデルの重みをロードし、パラメータを変異させて再開するロジックを実装。

### Step 2: Kaggle統合スクリプト (Kaggle Integration)
*   **参照**: [15. 結果スタッツ設計書](./15_Result_Stats_Spec.md)
*   **対象ファイル**: `scripts/kaggle_entry.py`
*   **タスク**:
    1.  Kaggle Dataset から前回のモデルとスタッツをロードする処理。
    2.  9時間の制限時間に合わせて学習ループを実行し、終了直前にモデルとログを保存する処理。
    3.  **stats_v{N}.bin** のバージョン管理ロジック。

---

### 付録: Python↔C++ バッチ推論 API（まとめ）

このリポジトリでは、C++ 側から Python モデルを効率よく呼び出すために 2 つのバッチ API をサポートします。

- `register_batch_inference(func)`
    - 旧来の行毎リスト（`list[list[float]]`）を受け取る方式。Python 側のシンプルな関数互換性を保つために残しています。
    - Python 関数は `(policies_list, values_list)` を返す必要があります。

- `register_batch_inference_numpy(func)` (flat / NumPy path)
    - C++ が連続したバッファ（`std::vector<float>`）を用意し、Python へ `numpy.ndarray`（shape=(batch, stride)）として渡します。
    - Python 関数は `(policies, values)` を返します。`policies` は `numpy.ndarray(float32)` または `list[list[float]]`、`values` は 1D ndarray または list を受け付けます。
    - 返却 ndarray は `float32` が高速で安全。`float64` もサポートして C++ 側で `float32` に変換します。

ライフタイムと安全性
- C++ 側は `std::shared_ptr<std::vector<float>>` を作り、その所有ポインタを `py::capsule` に格納して `py::array_t<float>` の `base` として渡します。これにより Python 側の ndarray が C++ のメモリを参照している間、メモリが保持されます。
- ただし、プログラム終了時やベンチ実行後には明示的に `clear_batch_inference_numpy()`（C++ binding）を呼んで登録を解除してください。これによりクロスランゲージの参照が残ることで起こる不具合を防げます。

使用例（簡易）

```python
import numpy as np
import dm_ai_module

def numpy_model(arr: np.ndarray):
        batch = arr.shape[0]
        policies = np.zeros((batch, dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE), dtype=np.float32)
        values = np.zeros((batch,), dtype=np.float32)
        return policies, values

dm_ai_module.register_batch_inference_numpy(numpy_model)
# NeuralEvaluator を使った評価を行う
dm_ai_module.clear_batch_inference_numpy()
```

ベンチ実行
- スクリプト: `python/tests/benchmark_batch_inference.py` を参照。各サブベンチ後に `clear_batch_inference*()` を呼んで GC しています。

注意点
- Python で `register_batch_inference_numpy` に登録する関数は、可能なら `float32` かつ C-contiguous な `numpy.ndarray` を返すようにしてください。そうすることで C++ 側の高速パスが有効になります。
- デバッグ出力は `AI_DEBUG` を有効にしてビルドしたときのみ出力されます。


