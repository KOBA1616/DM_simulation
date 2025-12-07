#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"
#include "../generic_card_system.hpp"
#include "../target_utils.hpp"
#include <algorithm>
#include <random>

namespace dm::engine {

    class SearchHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            using namespace dm::core;

            // SEARCH_DECK
            if (action.type == EffectActionType::SEARCH_DECK) {
                 EffectDef ed;
                 ed.trigger = TriggerType::NONE;
                 ed.condition = ConditionDef{"NONE", 0, ""};

                 ActionDef move_act;
                 move_act.type = EffectActionType::RETURN_TO_HAND;
                 if (action.destination_zone == "MANA_ZONE") {
                     move_act.type = EffectActionType::SEND_TO_MANA;
                 }

                 ActionDef shuffle_act;
                 shuffle_act.type = EffectActionType::SHUFFLE_DECK;

                 ed.actions = { move_act, shuffle_act };

                 ActionDef mod_action = action;
                 if (mod_action.filter.zones.empty()) {
                     mod_action.filter.zones = {"DECK"};
                 }
                 if (!mod_action.filter.owner.has_value()) {
                     mod_action.filter.owner = "SELF";
                 }
                 GenericCardSystem::select_targets(game_state, mod_action, source_instance_id, ed, execution_context);
                 return;
            }

            // SHUFFLE_DECK
            if (action.type == EffectActionType::SHUFFLE_DECK) {
                PlayerID controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
                Player& controller = game_state.players[controller_id];
                std::shuffle(controller.deck.begin(), controller.deck.end(), game_state.rng);
            }

            // SEARCH_DECK_BOTTOM
            if (action.type == EffectActionType::SEARCH_DECK_BOTTOM) {
                // Find controller
                PlayerID controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
                Player& controller = game_state.players[controller_id];

                int look = action.value1;
                // Variable Linking
                if (!action.input_value_key.empty() && execution_context.count(action.input_value_key)) {
                    look = execution_context[action.input_value_key];
                }
                if (look == 0) look = 1;

                std::vector<CardInstance> looked;
                for (int i = 0; i < look; ++i) {
                    if (controller.deck.empty()) break;
                    looked.push_back(controller.deck.back());
                    controller.deck.pop_back();
                }

                int chosen_idx = -1;
                for (size_t i = 0; i < looked.size(); ++i) {
                    const CardData* cd = CardRegistry::get_card_data(looked[i].card_id);
                    if (!cd) continue;

                    if (TargetUtils::is_valid_target(looked[i], *cd, action.filter, game_state, controller_id, controller_id)) {
                        chosen_idx = (int)i;
                        break;
                    }
                }

                if (chosen_idx != -1) {
                    controller.hand.push_back(looked[chosen_idx]);
                }

                for (int i = 0; i < (int)looked.size(); ++i) {
                    if (i == chosen_idx) continue;
                    controller.deck.insert(controller.deck.begin(), looked[i]);
                }
            }
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_id, std::map<std::string, int>& context) override {
             using namespace dm::core;

             // SEARCH_DECK legacy/fallback path
             if (action.type == EffectActionType::SEARCH_DECK) {
                 Player& active = game_state.get_active_player();
                 for (int tid : targets) {
                     auto it = std::find_if(active.deck.begin(), active.deck.end(),
                         [tid](const CardInstance& c){ return c.instance_id == tid; });
                     if (it != active.deck.end()) {
                         active.hand.push_back(*it);
                         active.deck.erase(it);
                     }
                 }
                 std::shuffle(active.deck.begin(), active.deck.end(), game_state.rng);
             }
        }
    };
}
