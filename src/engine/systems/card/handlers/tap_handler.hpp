#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/selection_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/selection_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"

namespace dm::engine {

    class TapHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;

            std::vector<int> targets;
            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
            } else if (ctx.action.target_choice != "SELECT") {
                // Auto-selection (e.g. ALL_ENEMY or generic filter)
                // Need to find targets similar to resolve()
                PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);

                std::vector<std::pair<PlayerID, Zone>> zones_to_check;
                if (ctx.action.filter.zones.empty()) {
                    zones_to_check.push_back({0, Zone::BATTLE});
                    zones_to_check.push_back({1, Zone::BATTLE});
                } else {
                     for (const auto& z : ctx.action.filter.zones) {
                        if (z == "BATTLE_ZONE") {
                            zones_to_check.push_back({0, Zone::BATTLE});
                            zones_to_check.push_back({1, Zone::BATTLE});
                        }
                         // Tap logic usually applies to Battle Zone.
                         // Mana tap is different logic usually.
                    }
                }

                for (const auto& [pid, zone] : zones_to_check) {
                    Player& p = ctx.game_state.players[pid];
                    if (zone == Zone::BATTLE) {
                         for (auto& card : p.battle_zone) {
                             if (!ctx.card_db.count(card.card_id)) continue;
                             const auto& def = ctx.card_db.at(card.card_id);
                             if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                                  targets.push_back(card.instance_id);
                             }
                         }
                    }
                }
            }

            if (targets.empty()) return;

            // Generate instructions
            nlohmann::json args;
            args["type"] = "TAP";
            // Pass vector of targets
            // But we don't have direct vector passing in JSON easily for arbitrary length?
            // Actually nlohmann supports it.
            // But wait, "target" arg in handle_modify expects single int or $variable.
            // Oh, I can check handle_modify in pipeline_executor.cpp
            // It iterates "targets" if it's a list?
            // "if (inst.args.contains("target")) ... targets.push_back..."
            // It supports single int or variable resolving to list.
            // It DOES NOT parse a JSON array directly in my previous read.
            // Let's re-read handle_modify.

            // handle_modify:
            // if (inst.args.contains("target")) { ... if (target_val.is_number()) ... }
            // It does NOT iterate array.

            // So I must emit one instruction per target.
            for (int t : targets) {
                 nlohmann::json args_i;
                 args_i["type"] = "TAP";
                 args_i["target"] = t;
                 ctx.instruction_buffer->emplace_back(InstructionOp::MODIFY, args_i);
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;
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
