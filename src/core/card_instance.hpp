#pragma once
#include "types.hpp"
#include <vector>
#include <string>

namespace dm::core {

    struct CardInstance {
        int instance_id = -1;
        CardID card_id = 0;
        PlayerID owner = 0;
        bool is_tapped = false;
        bool summoning_sickness = true;
        int turn_played = 0;
        bool is_face_down = false;

        // Modifiers
        int power_modifier = 0;
        std::vector<std::string> added_races;
        std::vector<std::string> added_keywords;

        // Evolution/Cross Gear
        std::vector<CardInstance> underlying_cards;

        CardInstance() = default;
        CardInstance(CardID cid, int inst_id, PlayerID oid) : instance_id(inst_id), card_id(cid), owner(oid) {}
    };

}
