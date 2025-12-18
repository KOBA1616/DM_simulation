#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/game_command/commands.hpp"

namespace dm::engine {

    class TapHandler : public IActionHandler {
    public:
        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;
            // Generate SELECT instruction if not resolved
            if (!ctx.targets && (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT")) {
                 // Delegate to default selection compilation (handled by EffectSystem::compile_action main loop usually)
                 // But if we are here, we might need to emit specific selection logic.
                 // For now, assume selection happens before or we emit a SELECT op.

                 // If compile is called, we should generate instructions.
                 // If targets are missing, emit SELECT.
                 // This requires mapping ActionDef filter to InstructionOp::SELECT args.
                 if (ctx.instruction_buffer) {
                     nlohmann::json select_args;
                     select_args["out"] = "$targets";
                     select_args["filter"] = ctx.action.filter;
                     select_args["count"] = ctx.action.value1 > 0 ? ctx.action.value1 : 1;
                     ctx.instruction_buffer->emplace_back(InstructionOp::SELECT, select_args);

                     // Then operate on selection
                     nlohmann::json mod_args;
                     mod_args["type"] = "TAP";
                     mod_args["target"] = "$targets";
                     ctx.instruction_buffer->emplace_back(InstructionOp::MODIFY, mod_args);
                     return;
                 }
            }

            // If targets provided or implicit (ALL)
            std::vector<int> target_ids;
            if (ctx.targets) {
                target_ids = *ctx.targets;
            } else {
                // Check for ALL_ENEMY / ALL_SELF
                // We can emit a SELECT with appropriate filter
                if (ctx.instruction_buffer) {
                    nlohmann::json select_args;
                    select_args["out"] = "$targets";
                    // Build filter for ALL
                    FilterDef f = ctx.action.filter;
                    if (ctx.action.target_choice == "ALL_ENEMY") f.owner = "OPPONENT";
                    else if (ctx.action.target_choice == "ALL_SELF") f.owner = "SELF";

                    if (f.zones.empty()) f.zones = {"BATTLE_ZONE"}; // Default for Tap

                    select_args["filter"] = f;
                    select_args["count"] = 999; // ALL
                    ctx.instruction_buffer->emplace_back(InstructionOp::SELECT, select_args);

                    nlohmann::json mod_args;
                    mod_args["type"] = "TAP";
                    mod_args["target"] = "$targets";
                    ctx.instruction_buffer->emplace_back(InstructionOp::MODIFY, mod_args);
                    return;
                }
            }

            // Direct targets
            if (ctx.instruction_buffer && !target_ids.empty()) {
                nlohmann::json mod_args;
                mod_args["type"] = "TAP";
                mod_args["target"] = target_ids;
                ctx.instruction_buffer->emplace_back(InstructionOp::MODIFY, mod_args);
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            // Legacy Support (Delegates to compile logic in future, keeping for safety)
            using namespace dm::core;
            if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 EffectDef ed;
                 ed.trigger = TriggerType::NONE;
                 ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                 ed.actions = { ctx.action };
                 EffectSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            if (ctx.action.target_choice == "ALL_ENEMY") {
                 int controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                 int enemy = 1 - controller_id;
                 for (auto& c : ctx.game_state.players[enemy].battle_zone) {
                     game_command::MutateCommand cmd(c.instance_id, game_command::MutateCommand::MutationType::TAP);
                     cmd.execute(ctx.game_state);
                 }
                 return;
            }

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
                }
            }

            for (const auto& [pid, zone] : zones_to_check) {
                Player& p = ctx.game_state.players[pid];
                if (zone == Zone::BATTLE) {
                     for (auto& card : p.battle_zone) {
                         if (!ctx.card_db.count(card.card_id)) continue;
                         const auto& def = ctx.card_db.at(card.card_id);
                         if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                              game_command::MutateCommand cmd(card.instance_id, game_command::MutateCommand::MutationType::TAP);
                              cmd.execute(ctx.game_state);
                         }
                     }
                }
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             if (!ctx.targets) return;
            for (int tid : *ctx.targets) {
                 game_command::MutateCommand cmd(tid, game_command::MutateCommand::MutationType::TAP);
                 cmd.execute(ctx.game_state);
            }
        }
    };
}
