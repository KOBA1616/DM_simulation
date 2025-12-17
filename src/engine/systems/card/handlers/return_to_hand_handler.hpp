#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/card/target_utils.hpp"
#include <algorithm>

namespace dm::engine {

    class ReturnToHandHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
             if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.actions = { ctx.action };
                 EffectSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            // Auto-Return Logic
            using namespace dm::core;
            PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);

            // Determine zones (Default Battle Zone)
            std::vector<std::pair<PlayerID, Zone>> zones_to_check;
            if (ctx.action.filter.zones.empty()) {
                zones_to_check.push_back({0, Zone::BATTLE});
                zones_to_check.push_back({1, Zone::BATTLE});
            } else {
                 for (const auto& z : ctx.action.filter.zones) {
                    if (z == "BATTLE_ZONE") {
                        zones_to_check.push_back({0, Zone::BATTLE});
                        zones_to_check.push_back({1, Zone::BATTLE});
                    }
                    if (z == "MANA_ZONE") {
                        zones_to_check.push_back({0, Zone::MANA});
                        zones_to_check.push_back({1, Zone::MANA});
                    }
                    if (z == "GRAVEYARD") {
                        zones_to_check.push_back({0, Zone::GRAVEYARD});
                        zones_to_check.push_back({1, Zone::GRAVEYARD});
                    }
                }
            }

            std::vector<int> targets_to_return;

            for (const auto& [pid, zone] : zones_to_check) {
                Player& p = ctx.game_state.players[pid];
                const std::vector<CardInstance>* card_list = nullptr;
                if (zone == Zone::BATTLE) card_list = &p.battle_zone;
                else if (zone == Zone::MANA) card_list = &p.mana_zone;
                else if (zone == Zone::GRAVEYARD) card_list = &p.graveyard;

                if (!card_list) continue;

                for (const auto& card : *card_list) {
                    if (!ctx.card_db.count(card.card_id)) continue;
                    const auto& def = ctx.card_db.at(card.card_id);

                    if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                         // Check Just Diver?
                         if (pid != controller_id) {
                              if (TargetUtils::is_protected_by_just_diver(card, def, ctx.game_state, controller_id)) continue;
                         }
                         targets_to_return.push_back(card.instance_id);
                    }
                }
            }

            // Apply Return
            for (int tid : targets_to_return) {
                for (auto &p : ctx.game_state.players) {
                    auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it != p.battle_zone.end()) {
                        ZoneUtils::on_leave_battle_zone(ctx.game_state, *it);
                        CardInstance moved_card = *it;
                        p.hand.push_back(moved_card);
                        p.battle_zone.erase(it);
                        EffectSystem::instance().check_mega_last_burst(ctx.game_state, moved_card, ctx.card_db);
                        break;
                    }
                    auto it_mana = std::find_if(p.mana_zone.begin(), p.mana_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it_mana != p.mana_zone.end()) {
                         p.hand.push_back(*it_mana);
                         p.mana_zone.erase(it_mana);
                         break;
                    }
                    auto it_grave = std::find_if(p.graveyard.begin(), p.graveyard.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it_grave != p.graveyard.end()) {
                         p.hand.push_back(*it_grave);
                         p.graveyard.erase(it_grave);
                         break;
                    }
                }
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;
             for (int tid : *ctx.targets) {
                for (auto &p : ctx.game_state.players) {
                    auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it != p.battle_zone.end()) {
                        // Cleanup hierarchy
                        ZoneUtils::on_leave_battle_zone(ctx.game_state, *it);

                        CardInstance moved_card = *it;
                        p.hand.push_back(moved_card);
                        p.battle_zone.erase(it);

                        EffectSystem::instance().check_mega_last_burst(ctx.game_state, moved_card, ctx.card_db);
                        break;
                    }
                    auto it_mana = std::find_if(p.mana_zone.begin(), p.mana_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it_mana != p.mana_zone.end()) {
                         p.hand.push_back(*it_mana);
                         p.mana_zone.erase(it_mana);
                         break;
                    }
                    auto it_grave = std::find_if(p.graveyard.begin(), p.graveyard.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it_grave != p.graveyard.end()) {
                         p.hand.push_back(*it_grave);
                         p.graveyard.erase(it_grave);
                         break;
                    }
                }
            }
        }
    };
}
