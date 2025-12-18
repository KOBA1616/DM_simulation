#pragma once
#include "core/game_state.hpp"
#include "core/action.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/flow/reaction_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "engine/systems/game_logic_system.hpp"

namespace dm::engine {

    class AttackHandler : public IActionHandler {
    public:
        // Implementing IActionHandler interface for consistency
        // Note: AttackHandler was previously a static helper.
        // We now allow it to be used as an Effect Handler if needed,
        // or just expose static compile logic if we want to migrate GameLogicSystem usage.

        void compile_action(const ResolutionContext& ctx) override {
            // Generate ATTACK action
            // Assuming this handler handles "Attack Target" effect
            using namespace dm::core;

            int source_id = ctx.source_instance_id;
            int target_id = -1;
            int target_player = -1;

            if (ctx.targets && !ctx.targets->empty()) {
                target_id = ctx.targets->front();
            } else {
                 // Logic to find target?
                 // Usually attack targets are set via Action args or UI.
                 // If this is an effect "This creature attacks X", we need logic.
                 // For now, assume straightforward mapping from context.
                 if (ctx.action.target_instance_id != -1) target_id = ctx.action.target_instance_id;
                 if (ctx.action.target_player != -1) target_player = ctx.action.target_player;
            }

            nlohmann::json args;
            args["type"] = "ATTACK";
            args["source_id"] = source_id;
            args["target_id"] = target_id;
            args["target_player"] = target_player;

            ctx.instruction_buffer->emplace_back(InstructionOp::GAME_ACTION, args);
        }

        void resolve(const ResolutionContext& ctx) override {
            std::vector<dm::core::Instruction> instructions;
            ResolutionContext compile_ctx = ctx;
            compile_ctx.instruction_buffer = &instructions;

            compile_action(compile_ctx);

            if (instructions.empty()) return;

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.execute(instructions, ctx.game_state, ctx.card_db);
        }

        // Keep static method for legacy/GameLogicSystem usage if needed?
        // Actually GameLogicSystem::handle_attack implements the logic.
        // This class is now an adapter to generate the instruction.
    };
}
