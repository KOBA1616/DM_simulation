#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <algorithm>
#include <random>

namespace dm::engine {

    class SearchHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            std::vector<Instruction> insts;
            ResolutionContext temp_ctx = ctx;
            temp_ctx.instruction_buffer = &insts;

            // Use compile for all handled types
            compile_action(temp_ctx);

            if (!insts.empty()) {
                auto pipeline = std::make_shared<dm::engine::systems::PipelineExecutor>();
                ctx.game_state.active_pipeline = pipeline;
                pipeline->execute(insts, ctx.game_state, ctx.card_db);
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;

             // Phase 2: Dispatch Command Directly
             if (ctx.action.type == EffectPrimitive::SEARCH_DECK) {
                 Zone dest = (ctx.action.destination_zone == "MANA_ZONE") ? Zone::MANA : Zone::HAND;

                 dm::engine::systems::PipelineExecutor pipeline;
                 pipeline.set_context_var("$targets", *ctx.targets);

                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = (dest == Zone::MANA) ? "MANA" : "HAND";
                 move.args["target"] = "$targets";

                 Instruction shuffle(InstructionOp::MODIFY);
                 shuffle.args["type"] = "SHUFFLE";
                 shuffle.args["target"] = "DECK";

                 pipeline.execute({move, shuffle}, ctx.game_state, ctx.card_db);
                 return;
             }

             dm::engine::systems::PipelineExecutor pipeline;
             pipeline.set_context_var("$targets", *ctx.targets);

             if (ctx.action.type == EffectPrimitive::SEARCH_DECK_BOTTOM) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "HAND";
                 move.args["target"] = "$targets";
                 pipeline.execute({move}, ctx.game_state, ctx.card_db);
             }
             else if (ctx.action.type == EffectPrimitive::SEND_TO_DECK_BOTTOM) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "DECK";
                 move.args["to_bottom"] = true;
                 move.args["target"] = "$targets";
                 pipeline.execute({move}, ctx.game_state, ctx.card_db);
             }
        }

        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            if (ctx.action.type == EffectPrimitive::SHUFFLE_DECK) {
                // Use Command!
                nlohmann::json args;
                dm::core::CommandDef cmd;
                cmd.type = dm::core::CommandType::SHUFFLE_DECK;
                // Owner is implicit or context
                // We can put CommandDef into Instruction args
                args["cmd"] = cmd;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "EXECUTE_COMMAND";
                ctx.instruction_buffer->push_back(inst);
            }
            else if (ctx.action.type == EffectPrimitive::SEARCH_DECK) {
                 // Use Command based flow?
                 // SEARCH_DECK involves SELECT -> MOVE -> SHUFFLE.
                 // We can use the SEARCH_DECK Command if it handles selection?
                 // Our CommandSystem implementation of SEARCH_DECK handles selection via filter.

                 nlohmann::json args;
                 dm::core::CommandDef cmd;
                 cmd.type = dm::core::CommandType::SEARCH_DECK;
                 cmd.target_filter = ctx.action.filter;
                 cmd.amount = (ctx.action.value1 > 0) ? ctx.action.value1 : 1;
                 cmd.to_zone = (ctx.action.destination_zone == "MANA_ZONE") ? "MANA" : "HAND";

                 // If default zones missing
                 if (cmd.target_filter.zones.empty()) {
                     cmd.target_filter.zones = {"DECK"};
                 }

                 // So we keep the Instruction flow for now as it supports SELECT (Query).
                 // But we can replace the Shuffle part with Command.

                 Instruction select(InstructionOp::SELECT);
                 select.args["filter"] = ctx.action.filter;
                 select.args["out"] = "$search_selection";
                 if (select.args["filter"]["zones"].empty()) {
                     select.args["filter"]["zones"] = std::vector<std::string>{"DECK"};
                 }
                 select.args["count"] = (ctx.action.value1 > 0) ? ctx.action.value1 : 1;
                 ctx.instruction_buffer->push_back(select);

                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = (ctx.action.destination_zone == "MANA_ZONE") ? "MANA" : "HAND";
                 move.args["target"] = "$search_selection";
                 ctx.instruction_buffer->push_back(move);

                 // Shuffle via Command
                 nlohmann::json sargs;
                 dm::core::CommandDef scmd;
                 scmd.type = dm::core::CommandType::SHUFFLE_DECK;
                 sargs["cmd"] = scmd;
                 Instruction shuffle(InstructionOp::GAME_ACTION, sargs);
                 shuffle.args["type"] = "EXECUTE_COMMAND";
                 ctx.instruction_buffer->push_back(shuffle);
            }
            else if (ctx.action.type == EffectPrimitive::SEARCH_DECK_BOTTOM) {
                // ... (Keep existing logic for now as it uses Buffer) ...
                int look_count = ctx.action.value1;
                if (look_count == 0) look_count = 1;

                Instruction move_to_buf(InstructionOp::MOVE);
                move_to_buf.args["to"] = "BUFFER";
                move_to_buf.args["target"] = "DECK_BOTTOM";
                move_to_buf.args["count"] = look_count;
                ctx.instruction_buffer->push_back(move_to_buf);

                Instruction select(InstructionOp::SELECT);
                select.args["filter"] = ctx.action.filter;
                select.args["filter"]["zones"] = std::vector<std::string>{"EFFECT_BUFFER"};
                select.args["out"] = "$search_selection";
                select.args["count"] = 1;
                ctx.instruction_buffer->push_back(select);

                Instruction move_sel(InstructionOp::MOVE);
                move_sel.args["to"] = "HAND";
                move_sel.args["target"] = "$search_selection";
                ctx.instruction_buffer->push_back(move_sel);

                Instruction select_rest(InstructionOp::SELECT);
                select_rest.args["filter"]["zones"] = std::vector<std::string>{"EFFECT_BUFFER"};
                select_rest.args["out"] = "$buffer_rest";
                select_rest.args["count"] = 999;
                ctx.instruction_buffer->push_back(select_rest);

                Instruction move_rest(InstructionOp::MOVE);
                move_rest.args["to"] = "DECK";
                move_rest.args["to_bottom"] = true;
                move_rest.args["target"] = "$buffer_rest";
                ctx.instruction_buffer->push_back(move_rest);
            }
            else if (ctx.action.type == EffectPrimitive::SEND_TO_DECK_BOTTOM) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "DECK";
                 move.args["to_bottom"] = true;

                 if (ctx.action.scope == TargetScope::TARGET_SELECT) {
                     Instruction select(InstructionOp::SELECT);
                     select.args["filter"] = ctx.action.filter;
                     select.args["out"] = "$sdb_selection";
                     select.args["count"] = (ctx.action.value1 > 0) ? ctx.action.value1 : 1;
                     ctx.instruction_buffer->push_back(select);
                     move.args["target"] = "$sdb_selection";
                 }
                 else if (!ctx.action.input_value_key.empty()) {
                     move.args["target"] = "$" + ctx.action.input_value_key;
                 } else {
                     Instruction select(InstructionOp::SELECT);
                     select.args["filter"] = ctx.action.filter;
                     select.args["out"] = "$auto_sdb_selection";
                     select.args["count"] = 999;
                     ctx.instruction_buffer->push_back(select);
                     move.args["target"] = "$auto_sdb_selection";
                 }
                 ctx.instruction_buffer->push_back(move);
            }
        }
    };
}
