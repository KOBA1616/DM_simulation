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
        std::vector<int> my_shields;
        std::vector<int> my_deck; // Added

        int enemy_shield_count = 5;
        std::vector<int> enemy_battle_zone;
        std::vector<int> enemy_deck; // Added
        bool enemy_can_use_trigger = false;
        bool loop_proof_mode = false;
    };
}
