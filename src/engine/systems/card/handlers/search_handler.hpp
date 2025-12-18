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
            compile(temp_ctx);

            if (!insts.empty()) {
                dm::engine::systems::PipelineExecutor pipeline;
                pipeline.execute(insts, ctx.game_state, ctx.card_db);
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;

             dm::engine::systems::PipelineExecutor pipeline;
             pipeline.set_context_var("$targets", *ctx.targets);

             if (ctx.action.type == EffectActionType::SEARCH_DECK) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = (ctx.action.destination_zone == "MANA_ZONE") ? "MANA" : "HAND";
                 move.args["target"] = "$targets";

                 Instruction shuffle(InstructionOp::MODIFY);
                 shuffle.args["type"] = "SHUFFLE";
                 shuffle.args["target"] = "DECK";

                 pipeline.execute({move, shuffle}, ctx.game_state, ctx.card_db);
             }
             else if (ctx.action.type == EffectActionType::SEARCH_DECK_BOTTOM) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "HAND";
                 move.args["target"] = "$targets";

                 pipeline.execute({move}, ctx.game_state, ctx.card_db);
             }
             else if (ctx.action.type == EffectActionType::SEND_TO_DECK_BOTTOM) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "DECK";
                 move.args["to_bottom"] = true;
                 move.args["target"] = "$targets";
                 pipeline.execute({move}, ctx.game_state, ctx.card_db);
             }
        }

        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            if (ctx.action.type == EffectActionType::SHUFFLE_DECK) {
                Instruction shuffle(InstructionOp::MODIFY);
                shuffle.args["type"] = "SHUFFLE";
                shuffle.args["target"] = "DECK";
                ctx.instruction_buffer->push_back(shuffle);
            }
            else if (ctx.action.type == EffectActionType::SEARCH_DECK) {
                 std::string var_name = "$search_selection_" + std::to_string(ctx.game_state.turn_number);

                 Instruction select(InstructionOp::SELECT);
                 select.args["filter"] = ctx.action.filter;
                 select.args["out"] = var_name;

                 // Default to DECK if not specified
                 if (ctx.action.filter.zones.empty()) {
                     select.args["filter"]["zones"] = std::vector<std::string>{"DECK"};
                 }

                 // Count (Optional?)
                 int count = ctx.action.value1;
                 if (count == 0) count = 1;
                 select.args["count"] = count;

                 ctx.instruction_buffer->push_back(select);

                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = (ctx.action.destination_zone == "MANA_ZONE") ? "MANA" : "HAND";
                 move.args["target"] = var_name;
                 ctx.instruction_buffer->push_back(move);

                 Instruction shuffle(InstructionOp::MODIFY);
                 shuffle.args["type"] = "SHUFFLE";
                 shuffle.args["target"] = "DECK";
                 ctx.instruction_buffer->push_back(shuffle);
            }
            else if (ctx.action.type == EffectActionType::SEARCH_DECK_BOTTOM) {
                int look_count = ctx.action.value1;
                if (look_count == 0) look_count = 1;
                std::string var_sel = "$search_bottom_sel_" + std::to_string(ctx.game_state.turn_number);
                std::string var_rest = "$search_bottom_rest_" + std::to_string(ctx.game_state.turn_number);

                // 1. Move Bottom N to Buffer
                Instruction move_to_buf(InstructionOp::MOVE);
                move_to_buf.args["to"] = "BUFFER";
                move_to_buf.args["target"] = "DECK_BOTTOM";
                move_to_buf.args["count"] = look_count;
                ctx.instruction_buffer->push_back(move_to_buf);

                // 2. Select from Buffer
                Instruction select(InstructionOp::SELECT);
                select.args["filter"] = ctx.action.filter;
                select.args["filter"]["zones"] = std::vector<std::string>{"EFFECT_BUFFER"};
                select.args["out"] = var_sel;
                select.args["count"] = 1; // Usually 1
                ctx.instruction_buffer->push_back(select);

                // 3. Move Selection to Hand
                Instruction move_sel(InstructionOp::MOVE);
                move_sel.args["to"] = "HAND";
                move_sel.args["target"] = var_sel;
                ctx.instruction_buffer->push_back(move_sel);

                // 4. Move Rest (from Buffer) to Deck Bottom
                Instruction select_rest(InstructionOp::SELECT);
                select_rest.args["filter"]["zones"] = std::vector<std::string>{"EFFECT_BUFFER"};
                select_rest.args["out"] = var_rest;
                select_rest.args["count"] = 999;
                ctx.instruction_buffer->push_back(select_rest);

                Instruction move_rest(InstructionOp::MOVE);
                move_rest.args["to"] = "DECK";
                move_rest.args["to_bottom"] = true;
                move_rest.args["target"] = var_rest;
                ctx.instruction_buffer->push_back(move_rest);
            }
            else if (ctx.action.type == EffectActionType::SEND_TO_DECK_BOTTOM) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "DECK";
                 move.args["to_bottom"] = true;

                 if (ctx.action.scope == TargetScope::TARGET_SELECT) {
                     std::string var_name = "$sdb_selection_" + std::to_string(ctx.game_state.turn_number);

                     Instruction select(InstructionOp::SELECT);
                     select.args["filter"] = ctx.action.filter;
                     select.args["out"] = var_name;
                     select.args["count"] = (ctx.action.value1 > 0) ? ctx.action.value1 : 1;
                     ctx.instruction_buffer->push_back(select);
                     move.args["target"] = var_name;
                 }
                 else if (!ctx.action.input_value_key.empty()) {
                     move.args["target"] = "$" + ctx.action.input_value_key;
                 } else {
                     std::string var_name = "$auto_sdb_selection_" + std::to_string(ctx.game_state.turn_number);

                     Instruction select(InstructionOp::SELECT);
                     select.args["filter"] = ctx.action.filter;
                     select.args["out"] = var_name;
                     select.args["count"] = 999;
                     ctx.instruction_buffer->push_back(select);
                     move.args["target"] = var_name;
                 }
                 ctx.instruction_buffer->push_back(move);
            }
        }
    };
}
