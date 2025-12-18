#pragma once
#include "types.hpp"
#include <vector>
#include <map>
#include <string>

namespace dm::core {

    // CardInstance definition
    // Usually in types.hpp or card.hpp, but here for now based on previous file structure
    // Actually, types.hpp did not have it.
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
        // ... more modifiers

        // Evolution/Cross Gear
        std::vector<CardInstance> underlying_cards;

        CardInstance() = default;
        CardInstance(CardID cid, PlayerID oid) : card_id(cid), owner(oid) {}
    };

    struct CardStats {
        int play_count = 0;
        int win_count = 0;
        int sum_cost_discount = 0; // Sum of cost reduction used
        int sum_early_usage = 0;   // Sum of (turn played < cost)
        float sum_win_contribution = 0.0f; // AI Heuristic sum
    };

    struct TurnStats {
        int played_without_mana = 0;
        int cards_drawn_this_turn = 0;
        int cards_discarded_this_turn = 0;
        int creatures_played_this_turn = 0;
        int spells_cast_this_turn = 0;
        int current_chain_depth = 0;
        // ...
    };

}
