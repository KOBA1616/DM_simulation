# 17. AI向け詳細実装指示書 (Detailed Implementation Instructions for AI)

本ドキュメントは、GitHub Copilot (GPT-5 mini相当) などのAIアシスタントが、コンテキストを最小限に抑えつつ、迷いなく実装を行うための超詳細な指示書である。
各タスクは「ファイルパス」「変更内容」「具体的なコードスニペット」で構成されている。

---

## Phase 3: AIコアの進化 (AI Core Evolution)

### Step 1: 結果スタッツ基盤の実装 (Result Stats)

#### 1.1 `src/core/card_stats.hpp` の作成
*   **目的**: カードの16次元スタッツを保持する構造体を定義する。
*   **内容**:
```cpp
#pragma once
#include <vector>
#include <cmath>

namespace dm::core {

    struct CardStats {
        // 実行回数
        long long play_count = 0;
        long long win_count = 0;

        // 16次元スタッツ蓄積用 (doubleで合計を持ち、最後に平均化する)
        double sum_early_usage = 0.0; // 0: Timing
        double sum_late_usage = 0.0;  // 1
        double sum_trigger_rate = 0.0;// 2
        double sum_cost_discount = 0.0;// 3
        
        double sum_hand_adv = 0.0;    // 4: Advantage
        double sum_board_adv = 0.0;   // 5
        double sum_mana_adv = 0.0;    // 6
        double sum_shield_dmg = 0.0;  // 7
        
        double sum_hand_var = 0.0;    // 8: Risk
        double sum_board_var = 0.0;   // 9
        double sum_survival_rate = 0.0;// 10
        double sum_effect_death = 0.0;// 11
        
        double sum_win_contribution = 0.0; // 12: Impact (Win Rate deviation)
        double sum_comeback_win = 0.0;     // 13
        double sum_finish_blow = 0.0;      // 14
        double sum_deck_consumption = 0.0; // 15

        // ベクトル化 (正規化)
        std::vector<float> to_vector() const {
            std::vector<float> vec(16, 0.0f);
            if (play_count == 0) return vec;

            double n = static_cast<double>(play_count);
            vec[0] = static_cast<float>(sum_early_usage / n);
            vec[1] = static_cast<float>(sum_late_usage / n);
            vec[2] = static_cast<float>(sum_trigger_rate / n);
            vec[3] = static_cast<float>(sum_cost_discount / n);
            
            vec[4] = static_cast<float>(sum_hand_adv / n);
            vec[5] = static_cast<float>(sum_board_adv / n);
            vec[6] = static_cast<float>(sum_mana_adv / n);
            vec[7] = static_cast<float>(sum_shield_dmg / n);
            
            vec[8] = static_cast<float>(sum_hand_var / n);
            vec[9] = static_cast<float>(sum_board_var / n);
            vec[10] = static_cast<float>(sum_survival_rate / n);
            vec[11] = static_cast<float>(sum_effect_death / n);
            
            vec[12] = static_cast<float>(sum_win_contribution / n);
            vec[13] = static_cast<float>(sum_comeback_win / n);
            vec[14] = static_cast<float>(sum_finish_blow / n);
            vec[15] = static_cast<float>(sum_deck_consumption / n);
            
            return vec;
        }
    };
}
```

#### 1.2 `src/core/game_state.hpp` の修正
*   **目的**: `GameState` にスタッツ管理用のマップを追加する。
*   **変更点**:
    *   `#include "card_stats.hpp"` を追加。
    *   `std::map<CardID, CardStats> global_card_stats;` をメンバ変数に追加。
    *   `std::vector<float> vectorize_card_stats(CardID cid) const;` メソッド宣言を追加。

#### 1.3 `src/engine/game_state.cpp` の修正 (または新規作成)
*   **目的**: スタッツ取得ロジックの実装。
*   **内容**:
```cpp
#include "../core/game_state.hpp"

namespace dm::core {
    std::vector<float> GameState::vectorize_card_stats(CardID cid) const {
        auto it = global_card_stats.find(cid);
        if (it != global_card_stats.end()) {
            return it->second.to_vector();
        }
        // データがない場合はゼロベクトル
        return std::vector<float>(16, 0.0f);
    }
}
```

---

### Step 2: 非公開領域推論の実装 (POMDP)

#### 2.1 `src/core/game_state.hpp` の修正
*   **目的**: 公開領域のスタッツ合計を保持する変数を追加。
*   **変更点**:
    *   `CardStats initial_deck_stats_sum;` (初期デッキの合計値)
    *   `CardStats visible_stats_sum;` (現在見えているカードの合計値)
    *   `int initial_deck_count = 40;`
    *   `int visible_card_count = 0;`
    *   `void on_card_reveal(CardID cid);` メソッド宣言を追加。
    *   `std::vector<float> get_library_potential() const;` メソッド宣言を追加。

#### 2.2 `src/engine/game_state.cpp` の修正
*   **目的**: 差分更新ロジックの実装。
*   **内容**:
```cpp
    void GameState::on_card_reveal(CardID cid) {
        auto it = global_card_stats.find(cid);
        if (it == global_card_stats.end()) return;

        const auto& stats = it->second;
        // CardStats に operator+= を実装するか、個別に加算する
        // ここでは簡易的に実装
        visible_stats_sum.sum_early_usage += stats.sum_early_usage / stats.play_count;
        // ... (全16次元を加算) ...
        
        visible_card_count++;
    }

    std::vector<float> GameState::get_library_potential() const {
        int remaining = initial_deck_count - visible_card_count;
        if (remaining <= 0) return std::vector<float>(16, 0.0f);

        std::vector<float> potential(16);
        // (Initial - Visible) / Remaining
        // 各次元について計算
        // ...
        return potential;
    }
```

---

## Phase 4: 高度な学習手法 (Advanced Training)

### Step 1: シナリオモードの実装

#### 1.1 `src/core/scenario_config.hpp` の作成
*   **目的**: シナリオ設定構造体の定義。
*   **内容**:
```cpp
#pragma once
#include <vector>
#include "types.hpp"

namespace dm::core {
    struct ScenarioConfig {
        int my_mana = 0;
        std::vector<int> my_hand_cards;
        std::vector<int> my_battle_zone;
        std::vector<int> my_mana_zone;
        std::vector<int> my_grave_yard;
        
        int enemy_shield_count = 5;
        std::vector<int> enemy_battle_zone;
        bool enemy_can_use_trigger = false;
        bool loop_proof_mode = false;
    };
}
```

#### 1.2 `src/engine/game_instance.hpp` の作成
*   **目的**: `GameState` と `PhaseManager` をラップし、Pythonから使いやすくする。
*   **内容**:
```cpp
#pragma once
#include "../core/game_state.hpp"
#include "../core/scenario_config.hpp"
#include "../engine/flow/phase_manager.hpp"
#include "../engine/card_system/card_registry.hpp"

namespace dm::engine {
    class GameInstance {
    public:
        core::GameState state;
        const std::map<core::CardID, core::CardDefinition>& card_db;

        GameInstance(uint32_t seed, const std::map<core::CardID, core::CardDefinition>& db) 
            : state(seed), card_db(db) {}

        void reset_with_scenario(const core::ScenarioConfig& config) {
            // 1. ゾーンのクリア
            state.players[0].hand.clear();
            state.players[0].battle_zone.clear();
            state.players[0].mana_zone.clear();
            // ...

            // 2. カード生成と配置
            int instance_id = 0;
            for (int cid : config.my_hand_cards) {
                state.players[0].hand.emplace_back(cid, instance_id++);
            }
            // ... (他のゾーンも同様)

            // 3. 状態の整合性確保
            state.turn_number = 5; // 適当なターン数
            state.active_player_id = 0;
            state.current_phase = core::Phase::MAIN;
        }
    };
}
```

#### 1.3 `src/python/bindings.cpp` の修正
*   **目的**: `GameInstance` と `ScenarioConfig` をPythonに公開する。
*   **変更点**:
    *   `py::class_<ScenarioConfig>(m, "ScenarioConfig")` を定義。
    *   `py::class_<GameInstance>(m, "GameInstance")` を定義し、`reset_with_scenario` を公開。

---

## Phase 5: 自律進化エコシステム

### Step 1: PBTワーカーの実装

#### 1.1 `python/training/pbt_worker.py` の作成
*   **目的**: ハイパーパラメータを変異させながら学習するワーカー。
*   **内容**:
```python
import json
import random
import torch
from .trainer import Trainer

class PBTWorker:
    def __init__(self, worker_id, config_path):
        self.worker_id = worker_id
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.trainer = Trainer(self.config)
        self.best_score = -float('inf')

    def step(self):
        # 1. 学習実行
        metrics = self.trainer.train_epoch()
        
        # 2. 評価
        score = self.evaluate()
        
        # 3. PBT Logic (Exploit & Explore)
        if score < self.get_population_bottom_20_percent_score():
            # 上位モデルをコピー
            best_worker = self.get_best_worker()
            self.trainer.load_state_dict(best_worker.trainer.state_dict())
            
            # ハイパーパラメータ変異
            self.mutate_hyperparams()
    
    def mutate_hyperparams(self):
        # 学習率を 0.8倍 または 1.2倍 にする
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        new_lr = current_lr * random.choice([0.8, 1.2])
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = new_lr
```

---

## 実装時の注意点
1.  **コンパイルエラー回避**: ヘッダーファイルのインクルード順序に注意する。`types.hpp` -> `constants.hpp` -> `card_stats.hpp` -> `game_state.hpp` の順。
2.  **Pythonバインディング**: `std::vector` などのSTLコンテナは `pybind11/stl.h` によって自動変換されるが、ポインタの所有権（`py::return_value_policy`）に注意する。
3.  **デバッグ**: `AI_DEBUG` マクロを使用して、詳細なログ出力を埋め込むこと。
