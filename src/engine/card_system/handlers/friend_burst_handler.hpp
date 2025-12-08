#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"
#include "../generic_card_system.hpp"
#include "core/card_def.hpp"
#include "../card_registry.hpp"
#include "../target_utils.hpp"
#include <iostream>

namespace dm::engine {

    class FriendBurstHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            (void)game_state; (void)action; (void)source_instance_id; (void)execution_context;
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) override {
            using namespace dm::core;

            // 1. Validate targets
            if (targets.empty()) return; // Optional pass

            int target_id = targets[0];
            CardInstance* target_creature = game_state.get_card_instance(target_id);
            if (!target_creature) return;

            // 2. Tap target
            target_creature->is_tapped = true;

            // 3. Resolve Spell Side
            // Get source card definition from Registry (to get effects)
            CardInstance* source_card = game_state.get_card_instance(source_instance_id);
            if (!source_card) return;

            const CardData* data = CardRegistry::get_card_data(source_card->card_id);
            if (data && data->spell_side) {
                const auto& spell_def = *data->spell_side;

                // Increment spell count
                game_state.turn_stats.spells_cast_this_turn++;

                // Execute effects
                for (const auto& effect : spell_def.effects) {
                    GenericCardSystem::resolve_effect_with_context(game_state, effect, source_instance_id, execution_context);
                }
            }

            (void)action; (void)card_db;
        }
    };
}
