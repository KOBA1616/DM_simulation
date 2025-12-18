#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/pipeline_executor.hpp"

namespace dm::engine {

    class MoveToUnderCardHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;
             std::vector<Instruction> insts;
             ResolutionContext temp_ctx = ctx;
             temp_ctx.instruction_buffer = &insts;
             compile(temp_ctx);
             dm::engine::systems::PipelineExecutor pipeline;
             pipeline.execute(insts, ctx.game_state, ctx.card_db);
        }

        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            if (ctx.action.type == EffectActionType::MOVE_TO_UNDER_CARD) {
                 Instruction attach(InstructionOp::GAME_ACTION);
                 attach.args["type"] = "ATTACH_CARD";

                 if (!ctx.action.input_value_key.empty()) {
                     attach.args["source_id"] = "$" + ctx.action.input_value_key;
                 }
                 ctx.instruction_buffer->push_back(attach);
            }
        }
    };
}
