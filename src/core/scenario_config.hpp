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

        int enemy_shield_count = 5;
        std::vector<int> enemy_shields; // Explicit shield content override
        int enemy_deck_id = 0;          // For future deck-based generation
        std::vector<int> enemy_battle_zone;
        bool enemy_can_use_trigger = false;
        bool loop_proof_mode = false;
    };
}
