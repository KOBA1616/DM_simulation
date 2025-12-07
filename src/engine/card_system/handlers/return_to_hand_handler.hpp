#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"
#include "../generic_card_system.hpp"
#include <algorithm>

namespace dm::engine {

    class ReturnToHandHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            using namespace dm::core;
             if (action.scope == TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
                 EffectDef ed;
                 ed.trigger = TriggerType::NONE;
                 ed.condition = ConditionDef{"NONE", 0, ""};
                 ed.actions = { action };
                 GenericCardSystem::select_targets(game_state, action, source_instance_id, ed, execution_context);
                 return;
            }

            // Handle ALL_ENEMY
            if (action.target_choice == "ALL_ENEMY") {
                 int controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
                 int enemy_idx = 1 - controller_id;
                 auto& bz = game_state.players[enemy_idx].battle_zone;
                 for (auto& c : bz) {
                     game_state.players[enemy_idx].hand.push_back(c);
                     game_state.players[enemy_idx].hand.back().is_tapped = false;
                     game_state.players[enemy_idx].hand.back().summoning_sickness = true;
                     game_state.players[enemy_idx].hand.back().power_mod = 0;
                 }
                 bz.clear();
            }
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_id, std::map<std::string, int>& context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) override {
             using namespace dm::core;
             for (int tid : targets) {
                bool found = false;
                // Check Battle Zones
                for (auto &p : game_state.players) {
                    auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const CardInstance& c){ return c.instance_id == tid; });
                    if (it != p.battle_zone.end()) {
                        p.hand.push_back(*it);
                        p.battle_zone.erase(it);
                        p.hand.back().is_tapped = false;
                        p.hand.back().summoning_sickness = true;
                        found = true;
                        break;
                    }
                }
                // Check Buffer
                if (!found) {
                    auto it = std::find_if(game_state.effect_buffer.begin(), game_state.effect_buffer.end(),
                        [tid](const CardInstance& c){ return c.instance_id == tid; });
                    if (it != game_state.effect_buffer.end()) {
                        Player& active = game_state.get_active_player();
                        active.hand.push_back(*it);
                        game_state.effect_buffer.erase(it);
                        active.hand.back().is_tapped = false;
                        active.hand.back().summoning_sickness = true;
                        found = true;
                    }
                }
                // Check Mana Zone
                if (!found) {
                    for (auto &p : game_state.players) {
                        auto it = std::find_if(p.mana_zone.begin(), p.mana_zone.end(),
                            [tid](const CardInstance& c){ return c.instance_id == tid; });
                        if (it != p.mana_zone.end()) {
                            p.hand.push_back(*it);
                            p.mana_zone.erase(it);
                            p.hand.back().is_tapped = false;
                            p.hand.back().summoning_sickness = true;
                            found = true;
                            break;
                        }
                    }
                }
            }
        }
    };
}
