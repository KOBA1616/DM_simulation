#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"
#include "../generic_card_system.hpp"
#include "../target_utils.hpp"
#include <algorithm>

namespace dm::engine {

    class BufferHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            using namespace dm::core;

            // Find controller
            PlayerID controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
            Player& controller = game_state.players[controller_id];

            int val1 = action.value1;
             if (!action.input_value_key.empty() && execution_context.count(action.input_value_key)) {
                val1 = execution_context[action.input_value_key];
             }
             if (val1 == 0) val1 = 1;

            if (action.type == EffectActionType::MEKRAID) {
                int look = val1;
                if (look == 1) look = 3;

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
                    CardInstance card = looked[chosen_idx];
                    game_state.effect_buffer.push_back(card);
                    game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, card.instance_id, controller.id);
                }

                for (int i = 0; i < (int)looked.size(); ++i) {
                    if (i == chosen_idx) continue;
                    controller.deck.insert(controller.deck.begin(), looked[i]);
                }
            } else if (action.type == EffectActionType::LOOK_TO_BUFFER) {
                int count = val1;
                std::vector<CardInstance>* source = nullptr;
                if (action.source_zone == "DECK" || action.source_zone.empty()) {
                    source = &controller.deck;
                } else if (action.source_zone == "HAND") {
                    source = &controller.hand;
                }

                if (source) {
                    for (int i = 0; i < count; ++i) {
                        if (source->empty()) break;
                        CardInstance c = source->back();
                        source->pop_back();
                        c.is_face_down = false;
                        game_state.effect_buffer.push_back(c);
                    }
                }
            } else if (action.type == EffectActionType::MOVE_BUFFER_TO_ZONE) {
                std::vector<CardInstance>* dest = nullptr;
                if (action.destination_zone == "DECK_BOTTOM") {
                    dest = &controller.deck;
                    for (auto& c : game_state.effect_buffer) {
                        controller.deck.insert(controller.deck.begin(), c);
                    }
                    game_state.effect_buffer.clear();
                }
                else if (action.destination_zone == "GRAVEYARD") {
                    dest = &controller.graveyard;
                    for (auto& c : game_state.effect_buffer) {
                        controller.graveyard.push_back(c);
                    }
                    game_state.effect_buffer.clear();
                }
                else if (action.destination_zone == "HAND") {
                    dest = &controller.hand;
                     for (auto& c : game_state.effect_buffer) {
                        controller.hand.push_back(c);
                    }
                    game_state.effect_buffer.clear();
                }
            }
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_id, std::map<std::string, int>& context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) override {
             using namespace dm::core;
             if (action.type == EffectActionType::PLAY_FROM_BUFFER) {
                 Player& active = game_state.get_active_player();
                 for (int tid : targets) {
                      auto it = std::find_if(game_state.effect_buffer.begin(), game_state.effect_buffer.end(),
                          [tid](const CardInstance& c){ return c.instance_id == tid; });

                      if (it != game_state.effect_buffer.end()) {
                          game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, tid, active.id);
                      }
                 }
             }
        }
    };
}
