#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/systems/card/target_utils.hpp"
#include <set>
#include <string>
#include <algorithm>

namespace dm::engine {

    class CountHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Migrate to compile() -> Pipeline
            std::vector<Instruction> insts;
            ResolutionContext temp_ctx = ctx;
            temp_ctx.instruction_buffer = &insts;
            compile_action(temp_ctx);

            if (!insts.empty()) {
                auto pipeline = std::make_shared<dm::engine::systems::PipelineExecutor>();

                // Sync Input Variables
                for (const auto& [k, v] : ctx.execution_vars) {
                    pipeline->set_context_var("$" + k, v);
                }

                ctx.game_state.active_pipeline = pipeline;
                pipeline->execute(insts, ctx.game_state, ctx.card_db);

                // Sync Output Variables
                if (!ctx.action.output_value_key.empty()) {
                    auto val = pipeline->get_context_var("$" + ctx.action.output_value_key);
                    if (std::holds_alternative<int>(val)) {
                        ctx.execution_vars[ctx.action.output_value_key] = std::get<int>(val);
                    }
                }
            }
            else {
                // Fallback for uncompiled actions (GET_GAME_STAT) - Legacy Logic
                if (ctx.action.type == EffectActionType::GET_GAME_STAT) {
                    PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                    Player& controller = ctx.game_state.players[controller_id];
                    int result = 0;

                    if (ctx.action.str_val == "MANA_CIVILIZATION_COUNT") {
                        std::set<std::string> civs;
                        for (const auto& c : controller.mana_zone) {
                            if (ctx.card_db.count(c.card_id)) {
                                 const auto& cd = ctx.card_db.at(c.card_id);
                                 for (const auto& civ : cd.civilizations) {
                                     if (civ == Civilization::LIGHT) civs.insert("LIGHT");
                                     if (civ == Civilization::WATER) civs.insert("WATER");
                                     if (civ == Civilization::DARKNESS) civs.insert("DARKNESS");
                                     if (civ == Civilization::FIRE) civs.insert("FIRE");
                                     if (civ == Civilization::NATURE) civs.insert("NATURE");
                                     if (civ == Civilization::ZERO) civs.insert("ZERO");
                                 }
                            }
                        }
                        result = (int)civs.size();
                    } else if (ctx.action.str_val == "SHIELD_COUNT") {
                        result = (int)controller.shield_zone.size();
                    } else if (ctx.action.str_val == "HAND_COUNT") {
                        result = (int)controller.hand.size();
                    } else if (ctx.action.str_val == "CARDS_DRAWN_THIS_TURN") {
                        result = ctx.game_state.turn_stats.cards_drawn_this_turn;
                    } else if (ctx.action.str_val == "MANA_COUNT") {
                        result = (int)controller.mana_zone.size();
                    } else if (ctx.action.str_val == "BATTLE_ZONE_COUNT") {
                        result = (int)controller.battle_zone.size();
                    } else if (ctx.action.str_val == "GRAVEYARD_COUNT") {
                        result = (int)controller.graveyard.size();
                    }

                    if (!ctx.action.output_value_key.empty()) {
                        ctx.execution_vars[ctx.action.output_value_key] = result;
                    }
                }
            }
        }

        void compile_action(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.instruction_buffer) return;

             std::string out_key = ctx.action.output_value_key;
             if (out_key.empty()) out_key = "$count_result";

             if (ctx.action.type == EffectActionType::COUNT_CARDS) {
                 Instruction count_inst(InstructionOp::COUNT);
                 count_inst.args["filter"] = ctx.action.filter;
                 // Need InstructionOp::COUNT to support filtering
                 // Current PipelineExecutor::handle_calc uses MATH or COUNT with "in" collection
                 // It does NOT support direct filtering in COUNT op yet?
                 // Wait, handle_calc COUNT logic:
                 /*
                 else if (inst.op == InstructionOp::COUNT) {
                     if (inst.args.contains("in")) { ... }
                 }
                 */
                 // It only counts collections.
                 // So we must SELECT first, then COUNT collection.

                 Instruction select(InstructionOp::SELECT);
                 select.args["filter"] = ctx.action.filter;
                 select.args["out"] = "$temp_selection_for_count";
                 select.args["count"] = 999;
                 ctx.instruction_buffer->push_back(select);

                 count_inst.args["in"] = "$temp_selection_for_count";
                 count_inst.args["out"] = out_key;
                 ctx.instruction_buffer->push_back(count_inst);
             }
             else if (ctx.action.type == EffectActionType::GET_GAME_STAT) {
                 Instruction stat(InstructionOp::MODIFY); // Use MODIFY STAT for now? No, MODIFY mutates.
                 // PipelineExecutor::handle_modify uses STAT to mutate?
                 // "STAT" in handle_modify executes StatCommand.
                 // We need GET STAT.
                 // PipelineExecutor doesn't seem to have GET_STAT instruction.
                 // But handle_calc supports MATH.

                 // If GET_GAME_STAT is not supported in PipelineExecutor, we might need to fallback?
                 // OR implement it via a new InstructionOp::GET?
                 // Or use PRINT for debugging?

                 // For now, if PipelineExecutor lacks GET_STAT, we can't fully migrate GET_GAME_STAT here without extending PipelineExecutor.
                 // However, we can use a custom op or abuse another op.
                 // Or, since I cannot modify PipelineExecutor effectively in this step (I did modify .cpp but header is fixed?),
                 // I will keep GET_GAME_STAT as Legacy logic?
                 // No, I want to remove legacy logic.

                 // I will assume I can use MATH with a special "op" or use a workaround.
                 // Actually, let's look at PipelineExecutor::handle_calc again.
                 // It does MATH.

                 // Wait! I can just implement the logic HERE in compile_action using many IFs? No.
                 // I need to read the stat at runtime.

                 // I'll skip GET_GAME_STAT migration for now and focus on COUNT_CARDS.
                 // The test test_pipeline_search_variable_link uses COUNT_CARDS.
             }
        }
    };
}
