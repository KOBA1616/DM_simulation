#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/pipeline_executor.hpp"

namespace dm::engine {

    class RevealHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;
             std::vector<Instruction> insts;
             ResolutionContext temp_ctx = ctx;
             temp_ctx.instruction_buffer = &insts;
             compile_action(temp_ctx);
             dm::engine::systems::PipelineExecutor pipeline;
             pipeline.execute(insts, ctx.game_state, ctx.card_db);
        }

        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            // Reveal top N cards of deck
            int count = ctx.action.value1;
            if (count == 0) count = 1;

            // Use BUFFER to simulate reveal (move to public zone temporarily)
            // "Reveal" usually implies showing to opponent but keeping in zone, OR moving to a "Reveal Zone".
            // Current engine pattern often uses EFFECT_BUFFER for complex manipulation.

            Instruction move_to_buf(InstructionOp::MOVE);
            move_to_buf.args["to"] = "BUFFER";
            move_to_buf.args["target"] = "DECK_TOP";
            move_to_buf.args["count"] = count;

            ctx.instruction_buffer->push_back(move_to_buf);

            // Print message (Optional)
            Instruction print(InstructionOp::PRINT);
            print.args["msg"] = "Revealed cards moved to Buffer";
            ctx.instruction_buffer->push_back(print);

            // Note: After reveal, usually something happens (add to hand, etc.).
            // This handler might just be "Reveal and Leave" or "Reveal and Process".
            // If it's part of a chain (Look -> Add -> Bottom), separate actions handle subsequent steps.
            // But if it is standalone "Reveal Top Card", it usually stays there?
            // "Reveal top card of deck." usually implies checking it.
            // If we move to BUFFER, we must move it back if nothing else happens?
            // Current `EffectPrimitive` includes `LOOK_AT_TOP_DECK`? No, `REVEAL_TOP_CARDS`.

            // If this is just "Reveal", we might need to know what to do next.
            // But typically this is used with Variable Linking (Reveal -> Filter -> Add).
            // So leaving in BUFFER is correct for subsequent actions to pick up from "EFFECT_BUFFER".
        }
    };
}
