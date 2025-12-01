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
