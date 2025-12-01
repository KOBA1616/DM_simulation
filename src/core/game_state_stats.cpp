#include "core/game_state.hpp"

namespace dm::core {

    void GameState::on_card_reveal(CardID cid) {
        auto it = global_card_stats.find(cid);
        // If we have no stats for this card, nothing to add
        if (it == global_card_stats.end()) return;

        const CardStats &cs = it->second;
        // If play_count is zero, treat averages as zero
        double denom = cs.play_count > 0 ? static_cast<double>(cs.play_count) : 1.0;

        // Add average contributions to visible_stats_sum (we store sums)
        visible_stats_sum.sum_early_usage += (cs.sum_early_usage / denom);
        visible_stats_sum.sum_late_usage += (cs.sum_late_usage / denom);
        visible_stats_sum.sum_trigger_rate += (cs.sum_trigger_rate / denom);
        visible_stats_sum.sum_cost_discount += (cs.sum_cost_discount / denom);

        visible_stats_sum.sum_hand_adv += (cs.sum_hand_adv / denom);
        visible_stats_sum.sum_board_adv += (cs.sum_board_adv / denom);
        visible_stats_sum.sum_mana_adv += (cs.sum_mana_adv / denom);
        visible_stats_sum.sum_shield_dmg += (cs.sum_shield_dmg / denom);

        visible_stats_sum.sum_hand_var += (cs.sum_hand_var / denom);
        visible_stats_sum.sum_board_var += (cs.sum_board_var / denom);
        visible_stats_sum.sum_survival_rate += (cs.sum_survival_rate / denom);
        visible_stats_sum.sum_effect_death += (cs.sum_effect_death / denom);

        visible_stats_sum.sum_win_contribution += (cs.sum_win_contribution / denom);
        visible_stats_sum.sum_comeback_win += (cs.sum_comeback_win / denom);
        visible_stats_sum.sum_finish_blow += (cs.sum_finish_blow / denom);
        visible_stats_sum.sum_deck_consumption += (cs.sum_deck_consumption / denom);

        visible_card_count++;
    }

    std::vector<float> GameState::vectorize_card_stats(CardID cid) const {
        auto it = global_card_stats.find(cid);
        if (it != global_card_stats.end()) return it->second.to_vector();
        return std::vector<float>(16, 0.0f);
    }

    std::vector<float> GameState::get_library_potential() const {
        int remaining = initial_deck_count - visible_card_count;
        if (remaining <= 0) return std::vector<float>(16, 0.0f);

        std::vector<float> potential(16, 0.0f);

        // Compute (initial - visible) / remaining
        potential[0] = static_cast<float>((initial_deck_stats_sum.sum_early_usage - visible_stats_sum.sum_early_usage) / remaining);
        potential[1] = static_cast<float>((initial_deck_stats_sum.sum_late_usage - visible_stats_sum.sum_late_usage) / remaining);
        potential[2] = static_cast<float>((initial_deck_stats_sum.sum_trigger_rate - visible_stats_sum.sum_trigger_rate) / remaining);
        potential[3] = static_cast<float>((initial_deck_stats_sum.sum_cost_discount - visible_stats_sum.sum_cost_discount) / remaining);

        potential[4] = static_cast<float>((initial_deck_stats_sum.sum_hand_adv - visible_stats_sum.sum_hand_adv) / remaining);
        potential[5] = static_cast<float>((initial_deck_stats_sum.sum_board_adv - visible_stats_sum.sum_board_adv) / remaining);
        potential[6] = static_cast<float>((initial_deck_stats_sum.sum_mana_adv - visible_stats_sum.sum_mana_adv) / remaining);
        potential[7] = static_cast<float>((initial_deck_stats_sum.sum_shield_dmg - visible_stats_sum.sum_shield_dmg) / remaining);

        potential[8] = static_cast<float>((initial_deck_stats_sum.sum_hand_var - visible_stats_sum.sum_hand_var) / remaining);
        potential[9] = static_cast<float>((initial_deck_stats_sum.sum_board_var - visible_stats_sum.sum_board_var) / remaining);
        potential[10] = static_cast<float>((initial_deck_stats_sum.sum_survival_rate - visible_stats_sum.sum_survival_rate) / remaining);
        potential[11] = static_cast<float>((initial_deck_stats_sum.sum_effect_death - visible_stats_sum.sum_effect_death) / remaining);

        potential[12] = static_cast<float>((initial_deck_stats_sum.sum_win_contribution - visible_stats_sum.sum_win_contribution) / remaining);
        potential[13] = static_cast<float>((initial_deck_stats_sum.sum_comeback_win - visible_stats_sum.sum_comeback_win) / remaining);
        potential[14] = static_cast<float>((initial_deck_stats_sum.sum_finish_blow - visible_stats_sum.sum_finish_blow) / remaining);
        potential[15] = static_cast<float>((initial_deck_stats_sum.sum_deck_consumption - visible_stats_sum.sum_deck_consumption) / remaining);

        return potential;
    }

}
