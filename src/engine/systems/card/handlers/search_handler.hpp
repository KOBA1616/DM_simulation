#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/pipeline_executor.hpp" // Added include
#include <algorithm>
#include <random>

namespace dm::engine {

    class SearchHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Migrate to compile() -> Pipeline
            std::vector<Instruction> insts;
            ResolutionContext temp_ctx = ctx;
            temp_ctx.instruction_buffer = &insts;

            // Only use compile logic if it handles the specific action type fully.
            // Currently compile handles: SEARCH_DECK, SEARCH_DECK_BOTTOM, SEND_TO_DECK_BOTTOM.
            // If action type is SHUFFLE_DECK, compile doesn't handle it?
            // Actually compile handles SHUFFLE via MODIFY.

            // However, SearchHandler also handles SHUFFLE_DECK directly in legacy resolve.
            // And compile() logic for SEARCH_DECK *includes* SHUFFLE.
            // Does compile() handle standalone SHUFFLE_DECK?
            // No, compile() checks specific types.

            bool handled = false;
            if (ctx.action.type == EffectActionType::SEARCH_DECK ||
                ctx.action.type == EffectActionType::SEARCH_DECK_BOTTOM ||
                ctx.action.type == EffectActionType::SEND_TO_DECK_BOTTOM) {

                compile(temp_ctx);
                if (!insts.empty()) {
                    dm::engine::systems::PipelineExecutor pipeline;
                    pipeline.execute(insts, ctx.game_state, ctx.card_db);
                    handled = true;
                }
            }

            if (!handled) {
                // Fallback or Handle SHUFFLE_DECK
                if (ctx.action.type == EffectActionType::SHUFFLE_DECK) {
                    // We can use Pipeline for Shuffle too.
                    Instruction shuffle(InstructionOp::MODIFY);
                    shuffle.args["type"] = "SHUFFLE";
                    shuffle.args["target"] = "DECK";
                    dm::engine::systems::PipelineExecutor pipeline;
                    pipeline.execute({shuffle}, ctx.game_state, ctx.card_db);
                }
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;

             // SEARCH_DECK - legacy resolve_with_targets moves card from deck to hand.
             // But compile() generates SELECT then MOVE.
             // If we are here, selection is done via legacy path (if resolve called select_targets).
             // But if we used compile() in resolve(), then select_targets is done via "SELECT" instruction in Pipeline.
             // Pipeline "SELECT" stores selection in context vars.
             // Pipeline "MOVE" uses context vars.
             // So resolve_with_targets is NOT called if we use Pipeline entirely!

             // UNLESS: Pipeline execution uses "SELECT_TARGET" action which triggers legacy flow?
             // No, PipelineExecutor::handle_select does selection internally and stores in context var.
             // It does NOT queue a pending effect.

             // Wait, if Pipeline does everything, resolve_with_targets is obsolete for the Pipeline path.
             // But GenericCardSystem calls resolve().
             // If resolve() runs Pipeline, and Pipeline completes the action, we are done.

             // However, GenericCardSystem might call resolve_with_targets if something ELSE triggered it?
             // Or if we mix legacy and new.
             // For now, we update resolve_with_targets to use Pipeline commands for robustness,
             // in case it's called by legacy paths (e.g. Shield Trigger logic that sets targets directly?)

             dm::engine::systems::PipelineExecutor pipeline;
             pipeline.set_context_var("$targets", *ctx.targets);

             if (ctx.action.type == EffectActionType::SEARCH_DECK) {
                 // Move targets to Hand, Shuffle
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "HAND"; // or MANA based on dest
                 move.args["target"] = "$targets";

                 Instruction shuffle(InstructionOp::MODIFY);
                 shuffle.args["type"] = "SHUFFLE";
                 shuffle.args["target"] = "DECK";

                 pipeline.execute({move, shuffle}, ctx.game_state, ctx.card_db);
             }
             else if (ctx.action.type == EffectActionType::SEARCH_DECK_BOTTOM) {
                 // Move targets to Hand
                 // Move REST of Buffer to Bottom
                 // How do we know "Rest of Buffer"?
                 // resolve_with_targets assumes complex state management.
                 // If we are using Pipeline, we should avoid this callback.
                 // But for legacy support:

                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "HAND";
                 move.args["target"] = "$targets";

                 // Move buffer rest
                 Instruction move_rest(InstructionOp::MOVE);
                 move_rest.args["to"] = "DECK";
                 move_rest.args["to_bottom"] = true;
                 move_rest.args["target"] = "BUFFER"; // Move ALL from buffer?
                 // PipelineExecutor::handle_move with target="BUFFER" needs to select all in buffer?
                 // Currently target logic supports list or "DECK_TOP/BOTTOM".
                 // It doesn't support "ALL_IN_ZONE".
                 // We can use SELECT from Buffer first.

                 Instruction select_rest(InstructionOp::SELECT);
                 select_rest.args["filter"]["zones"] = std::vector<std::string>{"EFFECT_BUFFER"};
                 select_rest.args["out"] = "$rest";
                 select_rest.args["count"] = 999;

                 move_rest.args["target"] = "$rest";

                 pipeline.execute({move, select_rest, move_rest}, ctx.game_state, ctx.card_db);
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

            if (ctx.action.type == EffectActionType::SEARCH_DECK) {
                 Instruction select(InstructionOp::SELECT);
                 select.args["filter"] = ctx.action.filter;
                 select.args["out"] = "$search_selection";
                 if (ctx.action.filter.zones.empty()) {
                     select.args["filter"]["zones"] = std::vector<std::string>{"DECK"};
                 }
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
            else if (ctx.action.type == EffectActionType::SEARCH_DECK_BOTTOM) {
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
                ctx.instruction_buffer->push_back(select);

                Instruction move_sel(InstructionOp::MOVE);
                move_sel.args["to"] = "HAND";
                move_sel.args["target"] = "$search_selection";
                ctx.instruction_buffer->push_back(move_sel);

                Instruction move_rest(InstructionOp::MOVE);
                move_rest.args["to"] = "DECK";
                move_rest.args["to_bottom"] = true;
                Instruction select_rest(InstructionOp::SELECT);
                select_rest.args["filter"]["zones"] = std::vector<std::string>{"EFFECT_BUFFER"};
                select_rest.args["out"] = "$buffer_rest";
                select_rest.args["count"] = 999;
                ctx.instruction_buffer->push_back(select_rest);

                move_rest.args["target"] = "$buffer_rest";
                ctx.instruction_buffer->push_back(move_rest);
            }
            else if (ctx.action.type == EffectActionType::SEND_TO_DECK_BOTTOM) {
                 Instruction move(InstructionOp::MOVE);
                 move.args["to"] = "DECK";
                 move.args["to_bottom"] = true;

                 if (ctx.action.scope == TargetScope::TARGET_SELECT) {
                     move.args["target"] = "$selection";
                 } else if (!ctx.action.input_value_key.empty()) {
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
