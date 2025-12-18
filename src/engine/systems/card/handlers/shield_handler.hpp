#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <algorithm>
#include <vector>
#include <set>

namespace dm::engine {

    class ShieldHandler : public IActionHandler {
    public:
        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;

            PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
            Player& controller = ctx.game_state.players[controller_id];

            if (ctx.action.type == EffectActionType::ADD_SHIELD) {
                // Determine source zone: HAND, GRAVEYARD, DECK (default)
                std::string source_zone = "DECK";
                if (ctx.action.source_zone == "HAND") source_zone = "HAND";
                else if (ctx.action.source_zone == "GRAVEYARD") source_zone = "GRAVEYARD";

                int count = ctx.action.value1;
                if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                    count = ctx.execution_vars.at(ctx.action.input_value_key);
                }
                if (count == 0) count = 1;

                std::vector<int> targets;
                if (source_zone == "DECK") {
                    // Use virtual target
                    nlohmann::json move_args;
                    move_args["target"] = "DECK_TOP";
                    move_args["count"] = count;
                    move_args["to"] = "SHIELD"; // Corrected from SHIELD_ZONE
                    ctx.instruction_buffer->emplace_back(InstructionOp::MOVE, move_args);

                    if (!ctx.action.output_value_key.empty()) {
                         int avail = (int)controller.deck.size();
                         int actual = std::min(count, avail);
                         nlohmann::json math_args;
                         math_args["op"] = "+";
                         math_args["lhs"] = actual;
                         math_args["rhs"] = 0;
                         math_args["out"] = ctx.action.output_value_key;
                         ctx.instruction_buffer->emplace_back(InstructionOp::MATH, math_args);
                    }
                    return;
                } else {
                    // Hand or Graveyard
                    const std::vector<CardInstance>* src_vec = nullptr;
                    if (source_zone == "HAND") src_vec = &controller.hand;
                    else if (source_zone == "GRAVEYARD") src_vec = &controller.graveyard;

                    if (src_vec) {
                        int avail = (int)src_vec->size();
                        int actual = std::min(count, avail);
                        // Take from back
                        for (int i = 0; i < actual; ++i) {
                             targets.push_back(src_vec->at(avail - 1 - i).instance_id);
                        }
                    }
                }

                for (int t : targets) {
                     nlohmann::json move_args;
                     move_args["target"] = t;
                     move_args["to"] = "SHIELD"; // Corrected from SHIELD_ZONE
                     ctx.instruction_buffer->emplace_back(InstructionOp::MOVE, move_args);
                }

                if (!ctx.action.output_value_key.empty()) {
                     nlohmann::json math_args;
                     math_args["op"] = "+";
                     math_args["lhs"] = (int)targets.size();
                     math_args["rhs"] = 0;
                     math_args["out"] = ctx.action.output_value_key;
                     ctx.instruction_buffer->emplace_back(InstructionOp::MATH, math_args);
                }

            } else if (ctx.action.type == EffectActionType::SEND_SHIELD_TO_GRAVE) {
                // Determine Targets
                std::vector<int> targets;

                if (ctx.targets && !ctx.targets->empty()) {
                    std::set<int> target_ids(ctx.targets->begin(), ctx.targets->end());
                     if (ctx.action.inverse_target) {
                         std::vector<int> players_to_check;
                         if (ctx.action.filter.owner.has_value()) {
                            std::string req = ctx.action.filter.owner.value();
                            if (req == "SELF") players_to_check.push_back(controller_id);
                            else if (req == "OPPONENT") players_to_check.push_back(1 - controller_id);
                            else if (req == "BOTH") { players_to_check.push_back(controller_id); players_to_check.push_back(1 - controller_id); }
                         } else {
                             players_to_check.push_back(0);
                             players_to_check.push_back(1);
                         }

                         for (int pid : players_to_check) {
                             const auto& p = ctx.game_state.players[pid];
                             for (const auto& s : p.shield_zone) {
                                 if (target_ids.find(s.instance_id) == target_ids.end()) {
                                     targets.push_back(s.instance_id);
                                 }
                             }
                         }
                     } else {
                         targets = *ctx.targets;
                     }
                } else if (ctx.action.target_choice != "SELECT") {
                     if (controller.shield_zone.empty()) return;
                     targets.push_back(controller.shield_zone.back().instance_id);
                }

                for (int t : targets) {
                     nlohmann::json move_args;
                     move_args["target"] = t;
                     move_args["to"] = "GRAVEYARD";
                     ctx.instruction_buffer->emplace_back(InstructionOp::MOVE, move_args);
                }
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (ctx.action.type == EffectActionType::SEND_SHIELD_TO_GRAVE &&
               (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") &&
               (!ctx.targets || ctx.targets->empty())) {
                     EffectDef ed;
                     ed.trigger = TriggerType::NONE;
                     ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                     ed.actions = { ctx.action };
                     EffectSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                     return;
            }

            std::vector<dm::core::Instruction> instructions;
            ResolutionContext compile_ctx = ctx;
            compile_ctx.instruction_buffer = &instructions;

            compile(compile_ctx);

            if (instructions.empty()) return;

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.execute(instructions, ctx.game_state, ctx.card_db);
        }
    };
}
