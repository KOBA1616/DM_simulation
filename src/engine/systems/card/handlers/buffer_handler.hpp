#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include <algorithm>

namespace dm::engine {

    class BufferHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Find controller
            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
            Player& controller = ctx.game_state.players[controller_id];

            int val1 = ctx.action.value1;
             if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                val1 = ctx.execution_vars[ctx.action.input_value_key];
             }
             if (val1 == 0) val1 = 1;

            if (ctx.action.type == EffectActionType::MEKRAID) {
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
                    if (!ctx.card_db.count(looked[i].card_id)) continue;
                    const auto& cd = ctx.card_db.at(looked[i].card_id);
                     if (TargetUtils::is_valid_target(looked[i], cd, ctx.action.filter, ctx.game_state, controller_id, controller_id)) {
                        chosen_idx = (int)i;
                        break;
                    }
                }

                if (chosen_idx != -1) {
                    CardInstance card = looked[chosen_idx];
                    controller.effect_buffer.push_back(card);
                    ctx.game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, card.instance_id, controller.id);
                }

                for (int i = 0; i < (int)looked.size(); ++i) {
                    if (i == chosen_idx) continue;
                    controller.deck.insert(controller.deck.begin(), looked[i]);
                }
            } else if (ctx.action.type == EffectActionType::LOOK_TO_BUFFER) {
                int count = val1;
                std::vector<CardInstance>* source = nullptr;
                if (ctx.action.source_zone == "DECK" || ctx.action.source_zone.empty()) {
                    source = &controller.deck;
                } else if (ctx.action.source_zone == "HAND") {
                    source = &controller.hand;
                }

                if (source) {
                    for (int i = 0; i < count; ++i) {
                        if (source->empty()) break;
                        CardInstance c = source->back();
                        source->pop_back();
                        c.is_face_down = false;
                        controller.effect_buffer.push_back(c);
                    }
                }
            } else if (ctx.action.type == EffectActionType::MOVE_BUFFER_TO_ZONE) {
                if (ctx.action.destination_zone == "DECK_BOTTOM") {
                    for (auto& c : controller.effect_buffer) {
                        controller.deck.insert(controller.deck.begin(), c);
                    }
                    controller.effect_buffer.clear();
                }
                else if (ctx.action.destination_zone == "GRAVEYARD") {
                    for (auto& c : controller.effect_buffer) {
                        controller.graveyard.push_back(c);
                    }
                    controller.effect_buffer.clear();
                }
                else if (ctx.action.destination_zone == "HAND") {
                     for (auto& c : controller.effect_buffer) {
                        controller.hand.push_back(c);
                    }
                    controller.effect_buffer.clear();
                }
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;

             if (ctx.action.type == EffectActionType::PLAY_FROM_BUFFER) {
                 Player& active = ctx.game_state.get_active_player();
                 // Use controller of effect source or active player?
                 // Usually active player's buffer.
                 // But PLAY_FROM_BUFFER implies we are playing from OUR buffer.

                 // If the effect source is opponent (e.g. they force us to play?), buffer would be ours.
                 // Let's check 'controller' derived from source_instance_id
                 PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                 Player& controller = ctx.game_state.players[controller_id];

                 for (int tid : *ctx.targets) {
                      auto it = std::find_if(controller.effect_buffer.begin(), controller.effect_buffer.end(),
                          [tid](const CardInstance& c){ return c.instance_id == tid; });

                      if (it != controller.effect_buffer.end()) {
                          ctx.game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, tid, active.id);
                      }
                 }
             }
        }
    };
}
