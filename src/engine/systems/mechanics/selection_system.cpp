#include "selection_system.hpp"
#include "engine/systems/effects/effect_system.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/infrastructure/data/card_registry.hpp"
#include <algorithm>

namespace dm::engine::mechanics {

    using namespace dm::core;

    std::vector<int> dm::engine::mechanics::SelectionSystem::select_targets(GameState& game_state, const ActionDef& action, int source_instance_id, const EffectDef& continuation, std::map<std::string, int>& execution_context) {
        PlayerID controller = dm::engine::effects::EffectSystem::get_controller(game_state, source_instance_id);

        // First, attempt auto-selection optimization: collect valid targets and
        // if the requested count >= available targets, resolve immediately.
        std::vector<int> valid_targets;
        FilterDef filter = action.filter;
        std::vector<Zone> zones;
        if (filter.zones.empty()) {
            zones = {Zone::BATTLE, Zone::HAND, Zone::MANA, Zone::SHIELD};
        } else {
            for (const auto& z_str : filter.zones) {
                if (z_str == "BATTLE_ZONE") zones.push_back(Zone::BATTLE);
                else if (z_str == "HAND") zones.push_back(Zone::HAND);
                else if (z_str == "MANA_ZONE") zones.push_back(Zone::MANA);
                else if (z_str == "SHIELD_ZONE") zones.push_back(Zone::SHIELD);
                else if (z_str == "GRAVEYARD") zones.push_back(Zone::GRAVEYARD);
                else if (z_str == "DECK") zones.push_back(Zone::DECK);
                else if (z_str == "EFFECT_BUFFER") zones.push_back(Zone::BUFFER);
            }
        }

        const auto& card_db = dm::engine::infrastructure::CardRegistry::get_all_definitions();

        for (PlayerID pid : {controller, static_cast<PlayerID>(1 - controller)}) {
            for (Zone z : zones) {
                std::vector<int> zone_indices;
                if (z == Zone::BUFFER) {
                    for (const auto& c : game_state.players[pid].effect_buffer) zone_indices.push_back(c.instance_id);
                } else {
                    zone_indices = game_state.get_zone(pid, z);
                }

                for (int instance_id : zone_indices) {
                    if (instance_id < 0) continue;
                    const auto* card_ptr = game_state.get_card_instance(instance_id);
                    if (!card_ptr && z == Zone::BUFFER) {
                        const auto& buf = game_state.players[pid].effect_buffer;
                        auto it = std::find_if(buf.begin(), buf.end(), [instance_id](const CardInstance& c){ return c.instance_id == instance_id; });
                        if (it != buf.end()) card_ptr = &(*it);
                    }
                    if (!card_ptr) continue;
                    const auto& card = *card_ptr;

                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        if (dm::engine::utils::TargetUtils::is_valid_target(card, def, filter, game_state, controller, pid)) {
                            valid_targets.push_back(instance_id);
                        }
                    } else if (card.card_id == 0) {
                        // allow generic dummy cards
                        if (dm::engine::utils::TargetUtils::is_valid_target(card, CardDefinition(), filter, game_state, controller, pid)) {
                            valid_targets.push_back(instance_id);
                        }
                    }
                }
            }
        }

        int num_needed = 1;
        if (action.filter.count.has_value()) num_needed = action.filter.count.value();
        if (!action.input_value_key.empty() && execution_context.count(action.input_value_key)) {
            num_needed = execution_context[action.input_value_key];
        }

        if ((int)valid_targets.size() <= 0) {
            // No valid targets: nothing to do
            return {};
        }

        if (num_needed >= (int)valid_targets.size()) {
            // Auto-resolve: select all valid targets (or as many as available)
            dm::engine::effects::EffectSystem::instance().resolve_effect_with_targets(game_state, continuation, valid_targets, source_instance_id, card_db, execution_context);
            return valid_targets;
        }

        PendingEffect pending(EffectType::NONE, source_instance_id, controller);
        pending.resolve_type = ResolveType::TARGET_SELECT;
        pending.filter = action.filter;

        if (pending.filter.zones.empty()) {
             if (action.target_choice == "ALL_ENEMY") {
                 pending.filter.owner = "OPPONENT";
                 pending.filter.zones = {"BATTLE_ZONE"};
             }
        }

        if (action.filter.count.has_value()) {
            pending.num_targets_needed = action.filter.count.value();
        } else {
            pending.num_targets_needed = 1;
        }

        if (!action.input_value_key.empty()) {
            if (execution_context.count(action.input_value_key)) {
                pending.num_targets_needed = execution_context[action.input_value_key];
            }
        }

        pending.optional = action.optional;
        pending.effect_def = continuation;
        pending.execution_context = execution_context;

        game_state.pending_effects.push_back(pending);

        return {};
    }

}
