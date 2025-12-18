#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <algorithm>

namespace dm::engine {

    class ReturnToHandHandler : public IActionHandler {
    public:
        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;

            std::vector<int> targets;

            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
            } else if (ctx.action.target_choice != "SELECT") {
                // Auto-Return Logic
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
                        if (z == "GRAVEYARD") {
                            zones_to_check.push_back({0, Zone::GRAVEYARD});
                            zones_to_check.push_back({1, Zone::GRAVEYARD});
                        }
                    }
                }

                for (const auto& [pid, zone] : zones_to_check) {
                    Player& p = ctx.game_state.players[pid];
                    const std::vector<CardInstance>* card_list = nullptr;
                    if (zone == Zone::BATTLE) card_list = &p.battle_zone;
                    else if (zone == Zone::MANA) card_list = &p.mana_zone;
                    else if (zone == Zone::GRAVEYARD) card_list = &p.graveyard;

                    if (!card_list) continue;

                    for (const auto& card : *card_list) {
                        if (!ctx.card_db.count(card.card_id)) continue;
                        const auto& def = ctx.card_db.at(card.card_id);

                        if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                             // Check Just Diver
                             if (pid != controller_id) {
                                  if (TargetUtils::is_protected_by_just_diver(card, def, ctx.game_state, controller_id)) continue;
                             }
                             targets.push_back(card.instance_id);
                        }
                    }
                }
            }

            if (targets.empty()) return;

            for (int t : targets) {
                 nlohmann::json move_args;
                 move_args["target"] = t;
                 move_args["to"] = "HAND";
                 ctx.instruction_buffer->emplace_back(InstructionOp::MOVE, move_args);
                 // Note: TransitionCommand handles "on_leave_battle_zone" and Mega Last Burst via ZoneUtils logic
            }
        }

        void resolve(const ResolutionContext& ctx) override {
             if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.actions = { ctx.action };
                 EffectSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
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
