#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"
#include "../generic_card_system.hpp"

namespace dm::engine {

    class TapHandler : public IActionHandler {
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

            if (action.target_choice == "ALL_ENEMY") {
                 int controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
                 int enemy = 1 - controller_id;
                 for (auto& c : game_state.players[enemy].battle_zone) {
                     c.is_tapped = true;
                 }
            }
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_id, std::map<std::string, int>& context) override {
            // Helper to find instance (duplicated)
             auto find_inst = [&](int instance_id) -> dm::core::CardInstance* {
                for (auto& p : game_state.players) {
                    for (auto& c : p.battle_zone) if (c.instance_id == instance_id) return &c;
                    for (auto& c : p.hand) if (c.instance_id == instance_id) return &c;
                    for (auto& c : p.mana_zone) if (c.instance_id == instance_id) return &c;
                    for (auto& c : p.shield_zone) if (c.instance_id == instance_id) return &c;
                    for (auto& c : p.graveyard) if (c.instance_id == instance_id) return &c;
                }
                return nullptr;
            };

            for (int tid : targets) {
                dm::core::CardInstance* inst = find_inst(tid);
                if (inst) inst->is_tapped = true;
            }
        }
    };
}
