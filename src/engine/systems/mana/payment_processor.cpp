#include "payment_processor.hpp"
#include <algorithm>
#include <set>

namespace dm::engine {

    bool PaymentProcessor::process_payment(
        dm::core::GameState& game_state,
        dm::core::Player& player,
        const PaymentRequirement& req,
        const PaymentContext& context
    ) {
        if (req.is_g_zero) {
            game_state.turn_stats.played_without_mana = true;
            return true;
        }

        bool played_without_mana = true;

        if (req.uses_hyper_energy) {
            if (context.creatures_to_tap.size() != (size_t)req.hyper_energy_count) {
                return false;
            }
            if (!pay_hyper_energy(game_state, player, req.hyper_energy_count, context.creatures_to_tap)) {
                return false;
            }
            // If Hyper Energy reduced cost to 0, played_without_mana remains true
        }

        if (req.final_mana_cost > 0) {
             played_without_mana = false;
             if (!pay_mana(game_state, player, req.final_mana_cost, req.required_civs, context.mana_cards_to_tap)) {
                 return false;
             }
        }

        if (played_without_mana) {
            game_state.turn_stats.played_without_mana = true;
        }

        return true;
    }

    bool PaymentProcessor::pay_mana(
        dm::core::GameState& game_state,
        dm::core::Player& player,
        int cost,
        const std::vector<dm::core::Civilization>& /*required_civs*/,
        const std::vector<dm::core::CardID>& mana_ids
    ) {
        if (mana_ids.size() < (size_t)cost) return false;

        for (auto id : mana_ids) {
            dm::core::CardInstance* card = game_state.get_card_instance(id);
            if (!card) return false;
            if (card->is_tapped) return false;
            if (game_state.card_owner_map[card->instance_id] != player.id) return false;

            card->is_tapped = true;
        }

        return true;
    }

    bool PaymentProcessor::pay_hyper_energy(
        dm::core::GameState& game_state,
        dm::core::Player& player,
        int creature_count,
        const std::vector<dm::core::CardID>& creature_ids
    ) {
        if (creature_ids.size() != (size_t)creature_count) return false;

        for (auto id : creature_ids) {
            dm::core::CardInstance* card = game_state.get_card_instance(id);
            if (!card) return false;
            if (card->is_tapped) return false;
            if (game_state.card_owner_map[card->instance_id] != player.id) return false;

            card->is_tapped = true;
        }

        return true;
    }

}
