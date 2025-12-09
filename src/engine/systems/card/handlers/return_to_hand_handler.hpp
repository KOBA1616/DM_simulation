#pragma once
#include "../effect_system.hpp"
#include "../../../../core/game_state.hpp"
#include "../generic_card_system.hpp"
#include "../../../utils/zone_utils.hpp"
#include <algorithm>

namespace dm::engine {

    class ReturnToHandHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
             if (action.scope == dm::core::TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.actions = { action };
                 GenericCardSystem::select_targets(game_state, action, source_instance_id, ed, execution_context);
                 return;
            }
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& /*action*/, const std::vector<int>& targets, int /*source_instance_id*/, std::map<std::string, int>& /*execution_context*/, const std::map<dm::core::CardID, dm::core::CardDefinition>& /*card_db*/) override {
             for (int tid : targets) {
                for (auto &p : game_state.players) {
                    auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it != p.battle_zone.end()) {
                        // Cleanup hierarchy
                        ZoneUtils::on_leave_battle_zone(game_state, *it);

                        p.hand.push_back(*it);
                        p.battle_zone.erase(it);
                        break;
                    }
                    // Could also be in mana or shields (rare for Bounce, but technically "Return to Hand" targets battle zone usually)
                    // If action.filter.zones includes MANA_ZONE...
                    auto it_mana = std::find_if(p.mana_zone.begin(), p.mana_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it_mana != p.mana_zone.end()) {
                         p.hand.push_back(*it_mana);
                         p.mana_zone.erase(it_mana);
                         break;
                    }
                }
            }
        }
    };
}
