#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/flow/reaction_system.hpp"
#include <vector>

namespace dm::engine {
    class BreakShieldHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Delegate selection
            if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 EffectDef ed;
                 ed.trigger = TriggerType::NONE;
                 ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                 ed.actions = { ctx.action };
                 EffectSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            int count = ctx.action.value1;
            if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                count = ctx.execution_vars[ctx.action.input_value_key];
            }
            if (count == 0) count = 1;

            std::vector<int> target_shield_ids;
            std::vector<PlayerID> target_players;
            PlayerID controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);

            if (ctx.action.filter.owner.has_value()) {
                std::string owner = ctx.action.filter.owner.value();
                if (owner == "SELF") target_players.push_back(controller);
                else if (owner == "OPPONENT") target_players.push_back(1 - controller);
                else if (owner == "BOTH") { target_players.push_back(controller); target_players.push_back(1 - controller); }
            } else {
                target_players.push_back(1 - controller);
            }

            for (PlayerID pid : target_players) {
                Player& p = ctx.game_state.players[pid];
                std::vector<int> valid_in_player;
                for (const auto& s : p.shield_zone) {
                    if (!ctx.card_db.count(s.card_id)) continue;
                    const auto& def = ctx.card_db.at(s.card_id);
                    if (TargetUtils::is_valid_target(s, def, ctx.action.filter, ctx.game_state, controller, pid)) {
                        valid_in_player.push_back(s.instance_id);
                    }
                }

                int to_break = std::min(count, (int)valid_in_player.size());
                for (int i = 0; i < to_break; ++i) {
                     target_shield_ids.push_back(valid_in_player[valid_in_player.size() - 1 - i]);
                }
            }

            execute_breaks(ctx.game_state, target_shield_ids, ctx.source_instance_id, ctx.card_db);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.targets) return;
            execute_breaks(ctx.game_state, *ctx.targets, ctx.source_instance_id, ctx.card_db);
        }

    private:
        void execute_breaks(dm::core::GameState& game_state, const std::vector<int>& shield_ids, int source_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            for (int shield_id : shield_ids) {
                // Find shield owner and verify existence
                PlayerID shield_owner = EffectSystem::get_controller(game_state, shield_id);
                // Note: get_controller relies on card_owner_map, which is stable.
                // But verify shield is still in shield zone (it might have moved if multiple breaks happened)

                Player& defender = game_state.players[shield_owner];
                auto it = std::find_if(defender.shield_zone.begin(), defender.shield_zone.end(),
                    [shield_id](const CardInstance& c){ return c.instance_id == shield_id; });

                if (it == defender.shield_zone.end()) continue;

                CardInstance shield_card = *it; // Copy state before move

                // 1. AT_BREAK_SHIELD Trigger (on breaker)
                EffectSystem::instance().resolve_trigger(game_state, TriggerType::AT_BREAK_SHIELD, source_id, card_db);

                // 2. Determine Destination and S-Trigger
                bool shield_burn = false;

                // Check Breaker for Shield Burn
                // The source might be in Battle Zone or Graveyard (if sacrificed).
                // We need to find the source instance to check keywords.
                const CardDefinition* source_def = nullptr;
                CardInstance* source_card_ptr = game_state.get_card_instance(source_id);
                if (source_card_ptr && card_db.count(source_card_ptr->card_id)) {
                    source_def = &card_db.at(source_card_ptr->card_id);
                    if (source_def->keywords.shield_burn) {
                        shield_burn = true;
                    }
                }

                bool is_trigger = false;
                if (!shield_burn && card_db.count(shield_card.card_id)) {
                     const auto& def = card_db.at(shield_card.card_id);
                     if (TargetUtils::has_keyword_simple(game_state, shield_card, def, "SHIELD_TRIGGER")) {
                         is_trigger = true;
                     }
                }

                // 3. Move Card (TransitionCommand)
                Zone dest_zone = shield_burn ? Zone::GRAVEYARD : Zone::HAND;

                TransitionCommand cmd(shield_id, Zone::SHIELD, dest_zone, shield_owner);
                cmd.execute(game_state);

                // 4. Post-Move Logic (S-Trigger Queue / Strike Back)
                if (!shield_burn) {
                     if (is_trigger) {
                         game_state.pending_effects.emplace_back(EffectType::SHIELD_TRIGGER, shield_id, shield_owner);
                     }
                     // Reaction Window: ON_SHIELD_ADD
                     ReactionSystem::check_and_open_window(game_state, card_db, "ON_SHIELD_ADD", shield_owner);
                } else {
                     EffectSystem::instance().resolve_trigger(game_state, TriggerType::ON_DESTROY, shield_id, card_db);
                }
            }
        }
    };
}
