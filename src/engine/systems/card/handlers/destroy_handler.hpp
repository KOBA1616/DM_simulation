#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include <algorithm>

namespace dm::engine {

    class DestroyHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.trigger = dm::core::TriggerType::NONE;
                 ed.condition = dm::core::ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                 ed.actions = { ctx.action };
                 GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            // Auto-destruction logic for non-selection scopes (e.g. ALL, SELF)
            // Iterate all potential targets, validate with TargetUtils, and destroy.
            using namespace dm::core;
            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
            int destroyed_count = 0;

            // Determine zones to check
            std::vector<std::pair<PlayerID, Zone>> zones_to_check;
            for (const auto& z : ctx.action.filter.zones) {
                if (z == "BATTLE_ZONE") {
                    zones_to_check.push_back({0, Zone::BATTLE});
                    zones_to_check.push_back({1, Zone::BATTLE});
                }
            }
            if (zones_to_check.empty()) {
                 // Default to battle zone if not specified for Destroy
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
                         // Additional check for protection (e.g., Just Diver)
                         if (pid != controller_id) {
                              if (TargetUtils::is_protected_by_just_diver(card, def, ctx.game_state, controller_id)) continue;
                         }

                         targets_to_destroy.push_back(card.instance_id);
                    }
                }
            }

            // Apply destruction
            for (int tid : targets_to_destroy) {
                for (auto &p : ctx.game_state.players) {
                    auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it != p.battle_zone.end()) {
                        ZoneUtils::on_leave_battle_zone(ctx.game_state, *it);
                        CardInstance moved_card = *it;
                        p.graveyard.push_back(moved_card);
                        p.battle_zone.erase(it);
                        destroyed_count++;

                        GenericCardSystem::check_mega_last_burst(ctx.game_state, moved_card, ctx.card_db);
                        break;
                    }
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
                bool found = false;
                for (auto &p : ctx.game_state.players) {
                    // 1. Check Top Cards (Elements)
                    auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it != p.battle_zone.end()) {
                        // Cleanup hierarchy before moving
                        ZoneUtils::on_leave_battle_zone(ctx.game_state, *it);

                        CardInstance moved_card = *it;
                        p.graveyard.push_back(moved_card);
                        p.battle_zone.erase(it);
                        destroyed_count++;

                        GenericCardSystem::check_mega_last_burst(ctx.game_state, moved_card, ctx.card_db);
                        found = true;
                        break;
                    }

                    // 2. Check Underlying Cards
                    if (!found) {
                        for (auto& top_card : p.battle_zone) {
                            auto u_it = std::find_if(top_card.underlying_cards.begin(), top_card.underlying_cards.end(),
                                [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });

                            if (u_it != top_card.underlying_cards.end()) {
                                // Found underlying card
                                CardInstance moved_card = *u_it;
                                p.graveyard.push_back(moved_card);
                                top_card.underlying_cards.erase(u_it);
                                destroyed_count++;

                                GenericCardSystem::check_mega_last_burst(ctx.game_state, moved_card, ctx.card_db);
                                found = true;
                                break;
                            }
                        }
                    }
                    if (found) break;
                }
            }
            if (!ctx.action.output_value_key.empty()) {
                ctx.execution_vars[ctx.action.output_value_key] = destroyed_count;
            }
        }
    };
}
