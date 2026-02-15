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
                // ctx.game_state.active_pipeline = pipeline; // Removed
                pipeline->execute(insts, ctx.game_state, ctx.card_db);
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;

             dm::engine::systems::PipelineExecutor pipeline;
             pipeline.set_context_var("$targets", *ctx.targets);

             if (ctx.action.type == EffectPrimitive::SEARCH_DECK) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = (ctx.action.destination_zone == "MANA_ZONE") ? "MANA" : "HAND";
                 move.args["target"] = "$targets";

                 Instruction shuffle(InstructionOp::MODIFY);
                 shuffle.args["type"] = "SHUFFLE";
                 shuffle.args["target"] = "DECK";

                 pipeline.execute({move, shuffle}, ctx.game_state, ctx.card_db);
             }
             else if (ctx.action.type == EffectPrimitive::SEARCH_DECK_BOTTOM) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "HAND";
                 move.args["target"] = "$targets";

                 // Note: SEARCH_DECK_BOTTOM implies we looked at N cards, picked M, and put rest back.
                 // In legacy flow, "rest" is handled by complex state.
                 // In pipeline flow, we assume "rest" logic was handled if using compile().
                 // If using resolve_with_targets, we might be in a mixed state.
                 // For now, we move selected targets.
                 // "Rest" logic is hard to infer here without context of what was looked at.
                 // But typically resolve_with_targets is called AFTER selection.
                 // If we use pipeline for everything, this shouldn't be reached.

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
                Instruction shuffle(InstructionOp::MODIFY);
                shuffle.args["type"] = "SHUFFLE";
                shuffle.args["target"] = "DECK";
                ctx.instruction_buffer->push_back(shuffle);
            }
            else if (ctx.action.type == EffectPrimitive::SEARCH_DECK) {
                 Instruction select(InstructionOp::SELECT);
                 select.args["filter"] = ctx.action.filter;
                 select.args["out"] = "$search_selection";

                 // Default to DECK if not specified
                 if (ctx.action.filter.zones.empty()) {
                     select.args["filter"]["zones"] = std::vector<std::string>{"DECK"};
                 } else {
                     // Ensure DECK is in zones? Usually implicit for SEARCH_DECK.
                 }

                 // Count (Optional?)
                 // Search usually "Up to 1".
                 // Logic for "Up to" is handled by UI/AI.
                 int count = ctx.action.value1;
                 if (count == 0) count = 1;
                 select.args["count"] = count;

                 ctx.instruction_buffer->push_back(select);

                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = (ctx.action.destination_zone == "MANA_ZONE") ? "MANA" : "HAND";
                 move.args["target"] = "$search_selection";
                 ctx.instruction_buffer->push_back(move);

                 Instruction shuffle(InstructionOp::MODIFY);
                 shuffle.args["type"] = "SHUFFLE";
                 shuffle.args["target"] = "DECK";
                 ctx.instruction_buffer->push_back(shuffle);
            }
            else if (ctx.action.type == EffectPrimitive::SEARCH_DECK_BOTTOM) {
                int look_count = ctx.action.value1;
                if (look_count == 0) look_count = 1;

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
                select.args["out"] = "$search_selection";
                select.args["count"] = 1; // Usually 1
                ctx.instruction_buffer->push_back(select);

                // 3. Move Selection to Hand
                Instruction move_sel(InstructionOp::MOVE);
                move_sel.args["to"] = "HAND";
                move_sel.args["target"] = "$search_selection";
                ctx.instruction_buffer->push_back(move_sel);

                // 4. Move Rest (from Buffer) to Deck Bottom
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
                     move.args["target"] = "$selection"; // Assuming pre-selection or handled via separate SELECT step
                     // Wait, if scope is TARGET_SELECT, we need a SELECT instruction first!
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
