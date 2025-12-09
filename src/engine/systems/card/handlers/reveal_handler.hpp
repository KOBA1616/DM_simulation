#pragma once
#include "../effect_system.hpp"
#include "../../../../core/game_state.hpp"
#include "../generic_card_system.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine {

    class RevealHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            // REVEAL_CARDS usually targets cards (via Filter) or Zone.
            // If scoped to self/targets, we assume targets are already selected or implicit.
            // But if it's "Reveal top card of deck", it might use Filter zones.

            // For now, let's assume it's just logging for POMDP/Humans.
            // "Reveal" means information becomes public.
            // In GameState, we have on_card_reveal(CardID).
            // But usually this action has targets.

            // If called without targets, maybe it reveals from source?
            // Unused parameters for now if stub.
            (void)game_state; (void)action; (void)source_instance_id; (void)execution_context;
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) override {
             // Iterate targets and reveal them.
             (void)action; (void)source_instance_id; (void)execution_context;

             for (int id : targets) {
                 dm::core::CardInstance* card = game_state.get_card_instance(id);
                 if (card) {
                     game_state.on_card_reveal(card->card_id);
                     // If needed, we can log this to a "revealed_this_turn" list or similar.
                     // For now, on_card_reveal updates the visible stats.
                 }
             }
             (void)card_db;
        }
    };
}
