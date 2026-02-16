#pragma once
#include "engine/systems/effects/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/effects/effect_system.hpp"
#include "engine/utils/target_utils.hpp"
#include <algorithm>

namespace dm::engine {

    class BufferHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            int val1 = ctx.action.value1;
            if (val1 == 0) val1 = 1;

            if (ctx.action.type == EffectPrimitive::LOOK_TO_BUFFER || ctx.action.type == EffectPrimitive::REVEAL_TO_BUFFER) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["target"] = "DECK_TOP";
                 if (!ctx.action.input_value_key.empty()) {
                     move.args["count"] = "$" + ctx.action.input_value_key;
                 } else {
                     move.args["count"] = val1;
                 }
                 move.args["to"] = "BUFFER";
                 ctx.instruction_buffer->push_back(move);
            }
            else if (ctx.action.type == EffectPrimitive::MOVE_BUFFER_TO_ZONE) {
                 Instruction move(InstructionOp::MOVE);

                 std::string sel_var = ctx.selection_var;
                 if (sel_var.empty() && !ctx.action.input_value_key.empty()) {
                     sel_var = "$" + ctx.action.input_value_key;
                 } else if (sel_var.empty()) {
                     // Select all from buffer if no input
                     Instruction select(InstructionOp::SELECT);
                     select.args["filter"]["zones"] = std::vector<std::string>{"EFFECT_BUFFER"};
                     select.args["out"] = "$buffer_all";
                     select.args["count"] = 0; // Select All
                     ctx.instruction_buffer->push_back(select);
                     sel_var = "$buffer_all";
                 }

                 move.args["target"] = sel_var;

                 std::string dest = "GRAVEYARD";
                 if (ctx.action.destination_zone == "HAND") dest = "HAND";
                 else if (ctx.action.destination_zone == "DECK_BOTTOM") dest = "DECK";
                 else if (ctx.action.destination_zone == "MANA_ZONE") dest = "MANA";
                 else if (ctx.action.destination_zone == "BATTLE_ZONE") dest = "BATTLE";
                 else if (ctx.action.destination_zone == "SHIELD_ZONE") dest = "SHIELD";

                 move.args["to"] = dest;
                 if (ctx.action.destination_zone == "DECK_BOTTOM") move.args["to_bottom"] = true;

                 ctx.instruction_buffer->push_back(move);
            }
            // MEKRAID and PLAY_FROM_BUFFER logic is more complex, skip for now as not required for this verification
        }

        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Find controller
            PlayerID controller_id = dm::engine::effects::EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
            Player& controller = ctx.game_state.players[controller_id];

            int val1 = ctx.action.value1;
             if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                val1 = ctx.execution_vars[ctx.action.input_value_key];
             }
             if (val1 == 0) val1 = 1;

            if (ctx.action.type == EffectPrimitive::MEKRAID) {
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
                     if (dm::engine::utils::TargetUtils::is_valid_target(looked[i], cd, ctx.action.filter, ctx.game_state, controller_id, controller_id)) {
                        chosen_idx = (int)i;
                        break;
                    }
                }

                if (chosen_idx != -1) {
                    CardInstance card = looked[chosen_idx];
                    controller.effect_buffer.push_back(card);
                    ctx.game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, card.instance_id, controller.id);
                    // Set origin to DECK for interference checks
                    ctx.game_state.pending_effects.back().execution_context["$origin"] = (int)Zone::DECK;
                }

                for (int i = 0; i < (int)looked.size(); ++i) {
                    if (i == chosen_idx) continue;
                    controller.deck.insert(controller.deck.begin(), looked[i]);
                }
            } else if (ctx.action.type == EffectPrimitive::LOOK_TO_BUFFER) {
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
                        c.is_face_down = true;
                        controller.effect_buffer.push_back(c);
                    }
                }
            } else if (ctx.action.type == EffectPrimitive::REVEAL_TO_BUFFER) {
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
            } else if (ctx.action.type == EffectPrimitive::MOVE_BUFFER_TO_ZONE) {
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

             // Logic: PLAY_FROM_BUFFER with specific targets.
             if (ctx.action.type == EffectPrimitive::PLAY_FROM_BUFFER) {
                 Player& active = ctx.game_state.players[ctx.game_state.active_player_id];

                 // Determine origin from action definition if available
                 int origin_zone = -1;
                 if (ctx.action.source_zone == "DECK") origin_zone = (int)Zone::DECK;
                 else if (ctx.action.source_zone == "HAND") origin_zone = (int)Zone::HAND;
                 else if (ctx.action.source_zone == "GRAVEYARD") origin_zone = (int)Zone::GRAVEYARD;
                 else if (ctx.action.source_zone == "MANA_ZONE") origin_zone = (int)Zone::MANA;

                 for (int tid : *ctx.targets) {
                      // Check P0
                      auto it0 = std::find_if(ctx.game_state.players[0].effect_buffer.begin(), ctx.game_state.players[0].effect_buffer.end(),
                          [tid](const CardInstance& c){ return c.instance_id == tid; });

                      if (it0 != ctx.game_state.players[0].effect_buffer.end()) {
                          ctx.game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, tid, active.id);
                          if (origin_zone != -1) {
                              ctx.game_state.pending_effects.back().execution_context["$origin"] = origin_zone;
                          }
                          continue;
                      }

                      // Check P1
                      auto it1 = std::find_if(ctx.game_state.players[1].effect_buffer.begin(), ctx.game_state.players[1].effect_buffer.end(),
                          [tid](const CardInstance& c){ return c.instance_id == tid; });

                      if (it1 != ctx.game_state.players[1].effect_buffer.end()) {
                          ctx.game_state.pending_effects.emplace_back(EffectType::INTERNAL_PLAY, tid, active.id);
                          if (origin_zone != -1) {
                              ctx.game_state.pending_effects.back().execution_context["$origin"] = origin_zone;
                          }
                      }
                 }
             }
        }
    };
}
