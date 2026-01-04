#include "core/game_state.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

namespace dm::core {

    void GameState::initialize_card_stats(const std::map<CardID, CardDefinition>& card_db, int deck_size) {
        // Optimized: DO NOT iterate entire card_db to populate global_card_stats.
        // This prevents massive memory bloat when cloning GameState during MCTS.
        // global_card_stats now only tracks *local* changes (Copy-On-Write logic via get_mutable_card_stats).
        // Historical stats should be loaded via load_card_stats_from_json into historical_card_stats.

        // Set deck size and reset visible aggregates
        initial_deck_count = deck_size;
        visible_card_count = 0;
        visible_stats_sum = CardStats{};

        // initial_deck_stats_sum remains zero until historical stats are loaded and compute_initial_deck_sums is called
        initial_deck_stats_sum = CardStats{};
    }

    bool GameState::load_card_stats_from_json(const std::string& filepath) {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) return false;

        nlohmann::json j;
        try {
            ifs >> j;
        } catch (...) {
            return false;
        }

        // Expect an array of objects
        if (!j.is_array()) return false;

        auto loaded_stats = std::make_shared<std::map<CardID, CardStats>>();

        for (const auto &item : j) {
            if (!item.contains("id")) continue;
            CardID cid = static_cast<CardID>(item["id"].get<int>());
            CardStats cs;
            cs.play_count = item.value("play_count", 0LL);

            // If averages provided, convert to sums by multiplying by play_count
            if (item.contains("averages") && item["averages"].is_array()) {
                auto arr = item["averages"];
                for (size_t k = 0; k < 16 && k < arr.size(); ++k) {
                    double v = arr[k].get<double>();
                    double add = v * static_cast<double>(cs.play_count);
                    switch (k) {
                        case 0: cs.sum_early_usage = add; break;
                        case 1: cs.sum_late_usage = add; break;
                        case 2: cs.sum_trigger_rate = add; break;
                        case 3: cs.sum_cost_discount = add; break;
                        case 4: cs.sum_hand_adv = add; break;
                        case 5: cs.sum_board_adv = add; break;
                        case 6: cs.sum_mana_adv = add; break;
                        case 7: cs.sum_shield_dmg = add; break;
                        case 8: cs.sum_hand_var = add; break;
                        case 9: cs.sum_board_var = add; break;
                        case 10: cs.sum_survival_rate = add; break;
                        case 11: cs.sum_effect_death = add; break;
                        case 12: cs.sum_win_contribution = add; break;
                        case 13: cs.sum_comeback_win = add; break;
                        case 14: cs.sum_finish_blow = add; break;
                        case 15: cs.sum_deck_consumption = add; break;
                    }
                }
            } else if (item.contains("sums") && item["sums"].is_array()) {
                auto arr = item["sums"];
                for (size_t k = 0; k < 16 && k < arr.size(); ++k) {
                    double v = arr[k].get<double>();
                    switch (k) {
                        case 0: cs.sum_early_usage = v; break;
                        case 1: cs.sum_late_usage = v; break;
                        case 2: cs.sum_trigger_rate = v; break;
                        case 3: cs.sum_cost_discount = v; break;
                        case 4: cs.sum_hand_adv = v; break;
                        case 5: cs.sum_board_adv = v; break;
                        case 6: cs.sum_mana_adv = v; break;
                        case 7: cs.sum_shield_dmg = v; break;
                        case 8: cs.sum_hand_var = v; break;
                        case 9: cs.sum_board_var = v; break;
                        case 10: cs.sum_survival_rate = v; break;
                        case 11: cs.sum_effect_death = v; break;
                        case 12: cs.sum_win_contribution = v; break;
                        case 13: cs.sum_comeback_win = v; break;
                        case 14: cs.sum_finish_blow = v; break;
                        case 15: cs.sum_deck_consumption = v; break;
                    }
                }
            }

            (*loaded_stats)[cid] = cs;
        }

        // Assign to shared pointer for read-only sharing
        historical_card_stats = loaded_stats;

        // Clear local stats as we start fresh from history
        global_card_stats.clear();

        return true;
    }

    void GameState::compute_initial_deck_sums(const std::vector<CardID>& deck_list) {
        // Reset
        initial_deck_stats_sum = CardStats{};
        initial_deck_count = static_cast<int>(deck_list.size());

        for (CardID cid : deck_list) {
            CardStats cs = get_card_stats(cid);

            // Only aggregate if we have valid historical play data
            if (cs.play_count <= 0) continue;

            double denom = static_cast<double>(cs.play_count);

            initial_deck_stats_sum.sum_early_usage += (cs.sum_early_usage / denom);
            initial_deck_stats_sum.sum_late_usage += (cs.sum_late_usage / denom);
            initial_deck_stats_sum.sum_trigger_rate += (cs.sum_trigger_rate / denom);
            initial_deck_stats_sum.sum_cost_discount += (cs.sum_cost_discount / denom);

            initial_deck_stats_sum.sum_hand_adv += (cs.sum_hand_adv / denom);
            initial_deck_stats_sum.sum_board_adv += (cs.sum_board_adv / denom);
            initial_deck_stats_sum.sum_mana_adv += (cs.sum_mana_adv / denom);
            initial_deck_stats_sum.sum_shield_dmg += (cs.sum_shield_dmg / denom);

            initial_deck_stats_sum.sum_hand_var += (cs.sum_hand_var / denom);
            initial_deck_stats_sum.sum_board_var += (cs.sum_board_var / denom);
            initial_deck_stats_sum.sum_survival_rate += (cs.sum_survival_rate / denom);
            initial_deck_stats_sum.sum_effect_death += (cs.sum_effect_death / denom);

            initial_deck_stats_sum.sum_win_contribution += (cs.sum_win_contribution / denom);
            initial_deck_stats_sum.sum_comeback_win += (cs.sum_comeback_win / denom);
            initial_deck_stats_sum.sum_finish_blow += (cs.sum_finish_blow / denom);
            initial_deck_stats_sum.sum_deck_consumption += (cs.sum_deck_consumption / denom);
        }
    }


    void GameState::on_card_reveal(CardID cid) {
        CardStats cs = get_card_stats(cid);
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
        return get_card_stats(cid).to_vector();
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
