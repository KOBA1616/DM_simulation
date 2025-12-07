#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"
#include "../generic_card_system.hpp"

namespace dm::engine {

    class ShieldHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            using namespace dm::core;

            // Need controller
            PlayerID controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
            Player& controller = game_state.players[controller_id];

            if (action.type == EffectActionType::ADD_SHIELD) {
                std::vector<CardInstance>* source = &controller.deck;
                if (action.source_zone == "HAND") source = &controller.hand;
                else if (action.source_zone == "GRAVEYARD") source = &controller.graveyard;

                if (!source->empty()) {
                    CardInstance c = source->back();
                    source->pop_back();
                    c.is_face_down = true;
                    controller.shield_zone.push_back(c);
                }
            } else if (action.type == EffectActionType::SEND_SHIELD_TO_GRAVE) {
                // If not targeted, send top shield (last one)
                if (action.scope != TargetScope::TARGET_SELECT && action.target_choice != "SELECT") {
                     if (!controller.shield_zone.empty()) {
                        CardInstance c = controller.shield_zone.back();
                        controller.shield_zone.pop_back();
                        controller.graveyard.push_back(c);
                    }
                } else {
                     EffectDef ed;
                     ed.trigger = TriggerType::NONE;
                     ed.condition = ConditionDef{"NONE", 0, ""};
                     ed.actions = { action };
                     GenericCardSystem::select_targets(game_state, action, source_instance_id, ed, execution_context);
                }
            }
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_id, std::map<std::string, int>& context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) override {
            using namespace dm::core;
             if (action.type == EffectActionType::SEND_SHIELD_TO_GRAVE) {
                 for (int tid : targets) {
                    for (auto &p : game_state.players) {
                         auto it = std::find_if(p.shield_zone.begin(), p.shield_zone.end(),
                            [tid](const CardInstance& c){ return c.instance_id == tid; });
                         if (it != p.shield_zone.end()) {
                             p.graveyard.push_back(*it);
                             p.shield_zone.erase(it);
                             break;
                         }
                    }
                }
             }
        }
    };
}
