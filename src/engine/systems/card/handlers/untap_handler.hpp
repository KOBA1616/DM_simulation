#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/selection_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/selection_system.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"

namespace dm::engine {

    class UntapHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;

            std::vector<int> targets;
            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
            } else if (ctx.action.target_choice != "SELECT") {
                // Auto-selection
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
                        if (z == "MANA_ZONE") {
                            zones_to_check.push_back({0, Zone::MANA});
                            zones_to_check.push_back({1, Zone::MANA});
                        }
                    }
                }

                for (const auto& [pid, zone] : zones_to_check) {
                    Player& p = ctx.game_state.players[pid];
                    const std::vector<CardInstance>* card_list = nullptr;
                    if (zone == Zone::BATTLE) card_list = &p.battle_zone;
                    else if (zone == Zone::MANA) card_list = &p.mana_zone;

                    if (!card_list) continue;

                     for (auto& card : *card_list) {
                         if (!ctx.card_db.count(card.card_id)) continue;
                         const auto& def = ctx.card_db.at(card.card_id);
                         if (dm::engine::utils::TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                              targets.push_back(card.instance_id);
                         }
                     }
                }
            }

            if (targets.empty()) return;

            for (int t : targets) {
                 nlohmann::json args_i;
                 args_i["type"] = "UNTAP";
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
