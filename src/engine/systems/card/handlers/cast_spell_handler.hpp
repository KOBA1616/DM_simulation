#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/pipeline_executor.hpp"

namespace dm::engine {

    class CastSpellHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Gather targets
            std::vector<int> targets;
            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
            } else if (ctx.action.scope == dm::core::TargetScope::SELF) {
                // If scope is SELF, target the source card itself
                targets.push_back(ctx.source_instance_id);
            } else {
                 // Fallback or implicit logic if needed, similar to PlayHandler
                 // For CastSpell, usually targets are provided via SELECT or context.
                 return;
            }

            PlayerID controller = ctx.game_state.active_player_id;
            // Attempt to determine controller from source
            if (ctx.source_instance_id >= 0 && ctx.source_instance_id < (int)ctx.game_state.card_owner_map.size()) {
                 controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
            }

            for (int target_id : targets) {
                // Refactored to use direct MOVE to STACK as PipelineExecutor now supports "STACK".
                // Previously used PLAY_CARD_INTERNAL workaround.
                nlohmann::json args;
                args["target"] = target_id;
                args["to"] = "STACK";

                // Note: cast_side metadata was passed in PLAY_CARD_INTERNAL but wasn't used by handle_play_card logic for simple moves.
                // If it's needed for resolution, it should be handled by whatever processes the card on stack (e.g. handle_resolve_play).

                ctx.instruction_buffer->emplace_back(InstructionOp::MOVE, args);
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            // If targets were provided through resolve_with_targets, treat as a normal resolve.
            resolve(ctx);
        }

        void resolve(const ResolutionContext& ctx) override {
            std::vector<dm::core::Instruction> instructions;
            ResolutionContext compile_ctx = ctx;
            compile_ctx.instruction_buffer = &instructions;

            compile_action(compile_ctx);

            if (instructions.empty()) return;

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.execute(instructions, ctx.game_state, ctx.card_db);
            // After moving to STACK, attempt to immediately resolve any stacked plays
            if (ctx.targets && !ctx.targets->empty()) {
                for (int tid : *ctx.targets) {
                    // Determine controller for this instance
                    dm::core::PlayerID controller = ctx.game_state.active_player_id;
                    if (tid >= 0 && (size_t)tid < ctx.game_state.card_owner_map.size()) {
                        controller = (dm::core::PlayerID)ctx.game_state.card_owner_map[tid];
                    }
                    // Resolve play from stack for this instance
                    dm::engine::systems::GameLogicSystem::resolve_play_from_stack(ctx.game_state, tid, 0, dm::core::SpawnSource::HAND_SUMMON, controller, ctx.card_db);
                }
            }
        }
    };
}
