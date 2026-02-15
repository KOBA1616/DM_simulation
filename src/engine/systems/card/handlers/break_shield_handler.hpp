#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/selection_system.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/systems/effects/reaction_system.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"
#include "engine/systems/director/game_logic_system.hpp"
#include "engine/utils/target_utils.hpp"
#include <vector>

namespace dm::engine {
    class BreakShieldHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // 1. Determine Count
            int count = ctx.action.value1;
            if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                count = ctx.execution_vars.at(ctx.action.input_value_key);
            }
            if (count == 0) count = 1;

            // 2. Determine Targets
            std::vector<int> target_shield_ids;

            if (ctx.targets && !ctx.targets->empty()) {
                // If targets pre-selected
                target_shield_ids = *ctx.targets;
            } else if (ctx.action.target_choice != "SELECT") {
                // Auto-select logic (e.g. "Break opponent's shield")
                // Copy logic from resolve()
                PlayerID controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                std::vector<PlayerID> target_players;

                if (ctx.action.filter.owner.has_value()) {
                    std::string owner = ctx.action.filter.owner.value();
                    if (owner == "SELF") target_players.push_back(controller);
                    else if (owner == "OPPONENT") target_players.push_back(1 - controller);
                    else if (owner == "BOTH") { target_players.push_back(controller); target_players.push_back(1 - controller); }
                } else {
                    target_players.push_back(1 - controller); // Default to opponent
                }

                for (PlayerID pid : target_players) {
                    Player& p = ctx.game_state.players[pid];
                    std::vector<int> valid_in_player;
                    for (const auto& s : p.shield_zone) {
                        if (!ctx.card_db.count(s.card_id)) continue;
                        const auto& def = ctx.card_db.at(s.card_id);
                        // Filter validation
                        // Note: Filter usually specifies zones=["SHIELD_ZONE"]. If empty, we might skip?
                        // Assuming implied SHIELD_ZONE if undefined?
                        FilterDef f = ctx.action.filter;
                        if (f.zones.empty()) f.zones = {"SHIELD_ZONE"};

                        if (dm::engine::utils::TargetUtils::is_valid_target(s, def, f, ctx.game_state, controller, pid)) {
                            valid_in_player.push_back(s.instance_id);
                        }
                    }

                    int to_break = std::min(count, (int)valid_in_player.size());
                    // Break from top (highest index) usually? Or logic defines specific order?
                    // Implementation used end -> begin.
                            for (int i = 0; i < to_break; ++i) {
                                target_shield_ids.push_back(valid_in_player[valid_in_player.size() - 1 - i]);
                            }
                }
            }

            // 3. Generate Instructions (emit a single BREAK_SHIELD with shields array)
            if (!target_shield_ids.empty()) {
                nlohmann::json args;
                args["type"] = "BREAK_SHIELD";
                args["source_id"] = ctx.source_instance_id;
                args["shields"] = target_shield_ids;
                ctx.instruction_buffer->emplace_back(InstructionOp::GAME_ACTION, args);
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

             // Delegate selection explicitly if needed
            if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 EffectDef ed;
                 ed.trigger = TriggerType::NONE;
                 ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                 ed.actions = { ctx.action };
                 SelectionSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            std::vector<dm::core::Instruction> instructions;
            ResolutionContext compile_ctx = ctx;
            compile_ctx.instruction_buffer = &instructions;

            compile_action(compile_ctx);

            if (instructions.empty()) return;

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.execute(instructions, ctx.game_state, ctx.card_db);
        }
    };
}
