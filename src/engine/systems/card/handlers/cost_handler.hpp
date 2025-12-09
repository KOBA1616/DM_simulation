#pragma once
#include "../effect_system.hpp"
#include "../../../../core/game_state.hpp"
#include "../generic_card_system.hpp"
#include "../../../effects/effect_resolver.hpp"

namespace dm::engine {

    class CostHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            using namespace dm::core;

            if (action.type == EffectActionType::COST_REFERENCE) {
                if (action.str_val == "FINISH_HYPER_ENERGY") {
                    // Logic implies we need targets (tapped creatures)
                     if (action.scope == TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
                         EffectDef ed;
                         ed.trigger = TriggerType::NONE;
                         ed.condition = ConditionDef{"NONE", 0, ""};
                         ed.actions = { action };
                         GenericCardSystem::select_targets(game_state, action, source_instance_id, ed, execution_context);
                     }
                }
            }
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_id, std::map<std::string, int>& /*context*/, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) override {
             using namespace dm::core;

             if (action.type == EffectActionType::COST_REFERENCE && action.str_val == "FINISH_HYPER_ENERGY") {
                 for (int tid : targets) {
                     for (auto &p : game_state.players) {
                          auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                             [tid](const CardInstance& c){ return c.instance_id == tid; });
                          if (it != p.battle_zone.end()) {
                              it->is_tapped = true;
                          }
                     }
                 }
                 int taps = action.value1;
                 if (taps == 0) taps = (int)targets.size();

                 int reduction = taps * 2;

                 PlayerID controller = GenericCardSystem::get_controller(game_state, source_id);
                 EffectResolver::resolve_play_from_stack(game_state, source_id, reduction, SpawnSource::HAND_SUMMON, controller, card_db);
             }
        }
    };
}
