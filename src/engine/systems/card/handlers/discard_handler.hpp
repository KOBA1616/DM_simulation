#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <algorithm>
#include <random>

namespace dm::engine {

    class DiscardHandler : public IActionHandler {
    public:
        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Logic to identify targets or perform random selection
            std::vector<int> targets;

            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
            } else if (ctx.action.target_choice != "SELECT") {
                // Determine implicit targets (Random or All)
                PlayerID target_pid = ctx.game_state.active_player_id;

                // Target Player logic
                if (ctx.action.target_player == "OPPONENT") target_pid = 1 - ctx.game_state.active_player_id;
                else if (ctx.action.target_player == "SELF") target_pid = ctx.game_state.active_player_id;
                else if (ctx.action.scope == TargetScope::PLAYER_OPPONENT) target_pid = 1 - ctx.game_state.active_player_id;
                else if (ctx.action.scope == TargetScope::PLAYER_SELF) target_pid = ctx.game_state.active_player_id;

                const Player& p = ctx.game_state.players[target_pid];
                std::vector<int> discard_candidates;

                // Filter candidates
                for (const auto& card : p.hand) {
                    if (!ctx.card_db.count(card.card_id)) continue;
                     const auto& def = ctx.card_db.at(card.card_id);
                     if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, ctx.source_instance_id, target_pid)) {
                         discard_candidates.push_back(card.instance_id);
                     }
                }

                int count = ctx.action.filter.count.value_or(1);
                if (ctx.action.value1 > 0) count = ctx.action.value1;

                if (ctx.action.filter.selection_mode == "ALL" || ctx.action.target_choice == "ALL") {
                    count = discard_candidates.size();
                }

                if (!discard_candidates.empty()) {
                    if (ctx.action.scope == TargetScope::RANDOM || ctx.action.target_choice == "RANDOM" || ctx.action.filter.selection_mode == "RANDOM") {
                        // We need to shuffle candidates.
                        // Can we do this in `compile`? Yes, assuming simulation context stability.
                        // Wait, `compile` generates instructions. Randomness should ideally be in Instruction logic
                        // if we want deterministic replay from instructions, OR we bake the random choice here.
                        // Baking here matches current architecture where "Resolution" decides outcomes.
                        // But PipelineExecutor has RNG too.
                        // For "Pure Command Generation", we usually generate a "RANDOM SELECT" instruction.
                        // But we don't have that yet.
                        // We will bake the choice here for now.
                        // Note: To support deterministic replay, we should use state.rng here.
                        // But `ctx` has `game_state` which has `rng`.

                        // We need to make a copy to shuffle
                        std::vector<int> shuffled = discard_candidates;
                        // const casting rng? GameState is non-const in context?
                        // ResolutionContext has `dm::core::GameState& game_state;` so yes.
                        std::shuffle(shuffled.begin(), shuffled.end(), ctx.game_state.rng);

                        int num = std::min((int)shuffled.size(), count);
                        for (int i = 0; i < num; ++i) targets.push_back(shuffled[i]);
                    } else {
                         // Default logic (e.g. ALL or First N)
                         int num = std::min((int)discard_candidates.size(), count);
                         for (int i = 0; i < num; ++i) targets.push_back(discard_candidates[i]);
                    }
                }
            }

            if (targets.empty()) {
                // If output variable needed, set to 0?
                // Pipeline supports `COUNT` instruction but that's for counting existing things.
                // We can set var manually via NOOP or just skip.
                return;
            }

            for (int t : targets) {
                nlohmann::json move_args;
                move_args["target"] = t;
                move_args["to"] = "GRAVEYARD";
                ctx.instruction_buffer->emplace_back(InstructionOp::MOVE, move_args);

                // Track stats?
                // Currently DiscardHandler logic does "is_tapped=false" which MOVE handles.
                // It does NOT explicitly trigger ON_DISCARD here, relying on TransitionCommand -> ZONE_ENTER Graveyard.
                // This matches Move logic.
            }

            // If output variable needed
            if (!ctx.action.output_value_key.empty()) {
                 // We can use MATH instruction to set variable to constant
                 nlohmann::json math_args;
                 math_args["op"] = "+";
                 math_args["lhs"] = (int)targets.size();
                 math_args["rhs"] = 0;
                 math_args["out"] = ctx.action.output_value_key;
                 ctx.instruction_buffer->emplace_back(InstructionOp::MATH, math_args);
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 EffectSystem::instance().delegate_selection(ctx);
                 return;
            }

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
