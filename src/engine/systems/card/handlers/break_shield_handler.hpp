#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/effects/effect_resolver.hpp"
#include <vector>

namespace dm::engine {
    class BreakShieldHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Check if we need to select targets first
            if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 EffectDef ed;
                 ed.trigger = TriggerType::NONE;
                 ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                 ed.actions = { ctx.action };
                 GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            int count = ctx.action.value1;
            // Variable linking support
            if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                count = ctx.execution_vars[ctx.action.input_value_key];
            }
            if (count == 0) count = 1;

            // Determine targets (shields)
            std::vector<int> target_shield_ids;

            // Identify target player(s) based on Filter Owner
            std::vector<PlayerID> target_players;
            PlayerID controller = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);

            if (ctx.action.filter.owner.has_value()) {
                std::string owner = ctx.action.filter.owner.value();
                if (owner == "SELF") target_players.push_back(controller);
                else if (owner == "OPPONENT") target_players.push_back(1 - controller);
                else if (owner == "BOTH") { target_players.push_back(controller); target_players.push_back(1 - controller); }
            } else {
                // Default to OPPONENT if no filter owner specified for Break Shield
                target_players.push_back(1 - controller);
            }

            // Gather all valid shields from target players
            for (PlayerID pid : target_players) {
                Player& p = ctx.game_state.players[pid];
                // For simplified "Break N shields", we take from back (top)
                // If filter has conditions (civilization etc), we filter.

                std::vector<int> valid_in_player;
                for (const auto& s : p.shield_zone) {
                    if (!ctx.card_db.count(s.card_id)) continue;
                    const auto& def = ctx.card_db.at(s.card_id);
                    if (TargetUtils::is_valid_target(s, def, ctx.action.filter, ctx.game_state, controller, pid)) {
                        valid_in_player.push_back(s.instance_id);
                    }
                }

                // If count is applied per player or total? usually per player for "Break 1 shield" on opponent.
                // We pick 'count' shields from valid_in_player (from back)
                int to_break = std::min(count, (int)valid_in_player.size());
                // Taking from back (end of vector) matches standard break order
                for (int i = 0; i < to_break; ++i) {
                     target_shield_ids.push_back(valid_in_player[valid_in_player.size() - 1 - i]);
                }
            }

            // Execute Breaks
            for (int shield_id : target_shield_ids) {
                // Find owner of shield
                PlayerID shield_owner = GenericCardSystem::get_controller(ctx.game_state, shield_id);

                Action break_action;
                break_action.type = ActionType::BREAK_SHIELD;
                break_action.source_instance_id = ctx.source_instance_id;
                break_action.target_instance_id = shield_id;
                break_action.target_player = shield_owner;

                EffectResolver::resolve_action(ctx.game_state, break_action, ctx.card_db);
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.targets) return;

            for (int shield_id : *ctx.targets) {
                PlayerID shield_owner = GenericCardSystem::get_controller(ctx.game_state, shield_id);

                Action break_action;
                break_action.type = ActionType::BREAK_SHIELD;
                break_action.source_instance_id = ctx.source_instance_id;
                break_action.target_instance_id = shield_id;
                break_action.target_player = shield_owner;

                EffectResolver::resolve_action(ctx.game_state, break_action, ctx.card_db);
            }
        }
    };
}
