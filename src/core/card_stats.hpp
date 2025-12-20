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
        int sum_late_usage = 0;
        float sum_win_contribution = 0.0f; // AI Heuristic sum

        // Extended stats for AI analysis
        float sum_trigger_rate = 0.0f;
        float sum_hand_adv = 0.0f;
        float sum_board_adv = 0.0f;
        float sum_mana_adv = 0.0f;
        float sum_shield_dmg = 0.0f;
        float sum_hand_var = 0.0f;
        float sum_board_var = 0.0f;
        float sum_survival_rate = 0.0f;
        float sum_effect_death = 0.0f;
        float sum_comeback_win = 0.0f;
        float sum_finish_blow = 0.0f;
        float sum_deck_consumption = 0.0f;

        void record_usage(int turn, bool is_trigger, int cost_diff) {
            (void)turn;
            play_count++;
            if (cost_diff > 0) sum_cost_discount += cost_diff;
            if (is_trigger) sum_trigger_rate += 1.0f;
            // Simplified logic for now
        }

        std::vector<float> to_vector() const {
             return {
                 (float)play_count, (float)win_count, (float)sum_cost_discount,
                 (float)sum_early_usage, (float)sum_late_usage, sum_win_contribution,
                 sum_trigger_rate, sum_hand_adv, sum_board_adv, sum_mana_adv,
                 sum_shield_dmg, sum_hand_var, sum_board_var, sum_survival_rate,
                 sum_effect_death, sum_comeback_win, sum_finish_blow, sum_deck_consumption
             };
        }
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
