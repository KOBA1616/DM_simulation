#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/game_command/commands.hpp"
#include <algorithm>

namespace dm::engine {

    class DestroyHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Delegate selection logic
            if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.trigger = dm::core::TriggerType::NONE;
                 ed.condition = dm::core::ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                 ed.actions = { ctx.action };
                 GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            // Auto-destruction logic
            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);

            // Determine zones (Default: Battle Zone)
            std::vector<std::pair<PlayerID, Zone>> zones_to_check;
            for (const auto& z : ctx.action.filter.zones) {
                if (z == "BATTLE_ZONE") {
                    zones_to_check.push_back({0, Zone::BATTLE});
                    zones_to_check.push_back({1, Zone::BATTLE});
                }
            }
            if (zones_to_check.empty()) {
                 zones_to_check.push_back({0, Zone::BATTLE});
                 zones_to_check.push_back({1, Zone::BATTLE});
            }

            std::vector<int> targets_to_destroy;

            for (const auto& [pid, zone] : zones_to_check) {
                Player& p = ctx.game_state.players[pid];
                const std::vector<CardInstance>* card_list = nullptr;
                if (zone == Zone::BATTLE) card_list = &p.battle_zone;

                if (!card_list) continue;

                for (const auto& card : *card_list) {
                    if (!ctx.card_db.count(card.card_id)) continue;
                    const auto& def = ctx.card_db.at(card.card_id);

                    if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                         if (pid != controller_id) {
                              if (TargetUtils::is_protected_by_just_diver(card, def, ctx.game_state, controller_id)) continue;
                         }
                         targets_to_destroy.push_back(card.instance_id);
                    }
                }
            }

            int destroyed_count = 0;
            for (int tid : targets_to_destroy) {
                if (execute_destroy(ctx.game_state, tid, ctx.card_db)) {
                    destroyed_count++;
                }
            }

            if (!ctx.action.output_value_key.empty()) {
                ctx.execution_vars[ctx.action.output_value_key] = destroyed_count;
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.targets) return;
            int destroyed_count = 0;
            for (int tid : *ctx.targets) {
                if (execute_destroy(ctx.game_state, tid, ctx.card_db)) {
                    destroyed_count++;
                }
            }
            if (!ctx.action.output_value_key.empty()) {
                ctx.execution_vars[ctx.action.output_value_key] = destroyed_count;
            }
        }

    private:
        bool execute_destroy(dm::core::GameState& game_state, int instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            // Find the card to capture state and underlying cards
            CardInstance card_copy;
            PlayerID owner_id = 0;
            bool found = false;
            bool is_underlying = false;
            CardInstance* parent_card = nullptr;

            for (auto& p : game_state.players) {
                // 1. Top-level cards
                auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                    [instance_id](const CardInstance& c){ return c.instance_id == instance_id; });

                if (it != p.battle_zone.end()) {
                    card_copy = *it;
                    owner_id = p.id;
                    found = true;
                    break;
                }

                // 2. Underlying cards
                for (auto& top : p.battle_zone) {
                    auto u_it = std::find_if(top.underlying_cards.begin(), top.underlying_cards.end(),
                        [instance_id](const CardInstance& c){ return c.instance_id == instance_id; });
                    if (u_it != top.underlying_cards.end()) {
                        card_copy = *u_it;
                        owner_id = p.id;
                        found = true;
                        is_underlying = true;
                        parent_card = &top;
                        break;
                    }
                }
                if (found) break;
            }

            if (!found) return false;

            // Check Eternal Omega before destroy
            // TODO: Move this to an Interceptor?
            // For now, keep as inline check or passive effect check

            if (!is_underlying) {
                ZoneUtils::on_leave_battle_zone(game_state, card_copy);
            }

            if (is_underlying && parent_card) {
                 auto u_it = std::find_if(parent_card->underlying_cards.begin(), parent_card->underlying_cards.end(),
                        [instance_id](const CardInstance& c){ return c.instance_id == instance_id; });
                 if (u_it != parent_card->underlying_cards.end()) {
                     parent_card->underlying_cards.erase(u_it);
                 }
                 game_state.players[owner_id].graveyard.push_back(card_copy);
                 if (game_state.card_owner_map.size() > (size_t)instance_id) game_state.card_owner_map[instance_id] = owner_id;

                 GenericCardSystem::check_mega_last_burst(game_state, card_copy, card_db);
                 return true;
            } else {
                // Standard Top-Level Destroy
                // TransitionCommand handles updating card_owner_map and moving vectors.
                auto cmd = std::make_unique<TransitionCommand>(instance_id, Zone::BATTLE, Zone::GRAVEYARD, owner_id);
                game_state.execute_command(std::move(cmd));

                // Post-move logic
                GenericCardSystem::check_mega_last_burst(game_state, card_copy, card_db);
                return true;
            }
        }
    };
}
