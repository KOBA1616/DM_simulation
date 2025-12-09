#pragma once
#include "../effect_system.hpp"
#include "../../../../core/game_state.hpp"
#include "../generic_card_system.hpp"
#include "../../../utils/zone_utils.hpp"
#include <algorithm>

namespace dm::engine {

    class DestroyHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            if (action.scope == dm::core::TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.trigger = dm::core::TriggerType::NONE;
                 ed.condition = dm::core::ConditionDef{"NONE", 0, ""};
                 // Recursion fix: The next action shouldn't specify selection again.
                 // Ideally we copy the action but change scope to NONE or something.
                 // For now, reliance on resolve_with_targets being called next is key.
                 ed.actions = { action };
                 GenericCardSystem::select_targets(game_state, action, source_instance_id, ed, execution_context);
                 return;
            }
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& /*action*/, const std::vector<int>& targets, int /*source_id*/, std::map<std::string, int>& /*context*/, const std::map<dm::core::CardID, dm::core::CardDefinition>& /*card_db*/) override {
            for (int tid : targets) {
                for (auto &p : game_state.players) {
                    auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it != p.battle_zone.end()) {
                        // Cleanup hierarchy before moving
                        ZoneUtils::on_leave_battle_zone(game_state, *it);

                        p.graveyard.push_back(*it);
                        p.battle_zone.erase(it);
                        // Trigger ON_DESTROY? (Handled by PhaseManager or Event System usually, but if manual move...)
                        // Currently GenericCardSystem triggers are checked elsewhere or added to queue.
                        break;
                    }
                }
            }
        }
    };
}
