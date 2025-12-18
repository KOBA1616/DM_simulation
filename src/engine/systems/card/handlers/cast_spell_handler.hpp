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
        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Gather targets
            std::vector<int> targets;
            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
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
                // Generate PLAY_CARD_INTERNAL (which maps to RESOLVE_PLAY with logic)
                // We assume PLAY_CARD_INTERNAL handles the "Move to Stack" + "Resolve" flow.
                // We add "cast_side" metadata for GameLogicSystem to handle spell sides.
                nlohmann::json args;
                args["type"] = "PLAY_CARD_INTERNAL";
                args["source_id"] = target_id;
                args["spawn_source"] = (int)SpawnSource::EFFECT_SUMMON;
                args["dest_override"] = 0; // Default
                args["target_player"] = controller;
                args["cast_side"] = ctx.action.cast_spell_side; // Metadata

                ctx.instruction_buffer->emplace_back(InstructionOp::GAME_ACTION, args);
            }
        }

        void resolve(const ResolutionContext& ctx) override {
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
