#pragma once
#include <vector>
#include <cmath>

namespace dm::core {

    struct CardStats {
        long long play_count = 0;
        long long win_count = 0;
        long long mana_source_count = 0; // Track how often used as mana

        double sum_early_usage = 0.0; // 0
        double sum_late_usage = 0.0;  // 1
        double sum_trigger_rate = 0.0;// 2
        double sum_cost_discount = 0.0;// 3

        double sum_hand_adv = 0.0;    // 4
        double sum_board_adv = 0.0;   // 5
        double sum_mana_adv = 0.0;    // 6
        double sum_shield_dmg = 0.0;  // 7

        double sum_hand_var = 0.0;    // 8
        double sum_board_var = 0.0;   // 9
        double sum_survival_rate = 0.0;// 10
        double sum_effect_death = 0.0;// 11

        double sum_win_contribution = 0.0; // 12
        double sum_comeback_win = 0.0;     // 13
        double sum_finish_blow = 0.0;      // 14
        double sum_deck_consumption = 0.0; // 15

        void record_usage(int turn, bool is_trigger, int cost_diff) {
            play_count++;
            if (turn <= 3) sum_early_usage += 1.0;
            if (turn >= 7) sum_late_usage += 1.0;
            if (is_trigger) sum_trigger_rate += 1.0;
            sum_cost_discount += static_cast<double>(cost_diff);
        }

        void record_mana_source() {
            mana_source_count++;
        }

        std::vector<float> to_vector() const {
            std::vector<float> vec(16, 0.0f);

            if (play_count == 0 && mana_source_count == 0) return vec;
            double n = static_cast<double>(play_count > 0 ? play_count : 1); // Avoid div by zero

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

        // helper: add another CardStats (component-wise sums)
        void add_from_avg(const CardStats &other_avg) {
            // other_avg is expected to hold averages (not sums), add to sums by incrementing sums by value
            sum_early_usage += other_avg.sum_early_usage;
            sum_late_usage += other_avg.sum_late_usage;
            sum_trigger_rate += other_avg.sum_trigger_rate;
            sum_cost_discount += other_avg.sum_cost_discount;

            sum_hand_adv += other_avg.sum_hand_adv;
            sum_board_adv += other_avg.sum_board_adv;
            sum_mana_adv += other_avg.sum_mana_adv;
            sum_shield_dmg += other_avg.sum_shield_dmg;

            sum_hand_var += other_avg.sum_hand_var;
            sum_board_var += other_avg.sum_board_var;
            sum_survival_rate += other_avg.sum_survival_rate;
            sum_effect_death += other_avg.sum_effect_death;

            sum_win_contribution += other_avg.sum_win_contribution;
            sum_comeback_win += other_avg.sum_comeback_win;
            sum_finish_blow += other_avg.sum_finish_blow;
            sum_deck_consumption += other_avg.sum_deck_consumption;
        }
    };
}
