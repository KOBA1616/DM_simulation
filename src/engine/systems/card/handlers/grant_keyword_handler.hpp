#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"
#include <iostream>

namespace dm::engine {

    class GrantKeywordHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // GrantKeywordHandler applies a PASSIVE effect via instruction.
            // But PipelineExecutor currently handles Instructions like MODIFY for atomic changes.
            // It does not natively support "ADD_PASSIVE_EFFECT" instruction type in my previous read.
            // Let's check PipelineExecutor.cpp handle_modify.
            // "ADD_KEYWORD" mutation type exists!
            // MutateCommand::MutationType::ADD_KEYWORD adds a keyword modifier to a specific card instance.
            // This is "Apply Modifier" pattern, effectively.

            // So if we have specific targets (from filter or context), we can apply ADD_KEYWORD mutation to them.
            // This mutation lasts until... MutateCommand usually applies it permanently unless handled by Modifier cleanup system.
            // The `GameState` has `active_modifiers` which track mutations with duration.
            // Wait, MutateCommand modifies CardInstance directly.
            // `active_modifiers` is separate.

            // If the effect is "Until end of turn", we should use `APPLY_MODIFIER` (Generic Modifier).
            // But GrantKeywordHandler specifically mentions "PassiveEffect" in old code.
            // PassiveEffect is global logic.

            // The user listed "GrantKeywordHandler" in Priority 3.
            // "Migration: InstructionOp::MODIFY (type: "ADD_KEYWORD")".
            // This implies using the atomic modification on targets.

            // So logic:
            // 1. Find targets (Filter or Context)
            // 2. Emit MODIFY (ADD_KEYWORD) for each.

            // NOTE: If using ADD_KEYWORD mutation, we need to ensure it expires if needed.
            // The current engine handles modifier expiration if registered in `active_modifiers`.
            // MutateCommand::MutationType::ADD_KEYWORD usually just sets the bit in `CardInstance`.
            // It does NOT add to `active_modifiers`.
            // So "Until end of turn" behavior relies on something else cleaning it up, or using a different command.

            // However, based on the instruction "InstructionOp::MODIFY (type: "ADD_KEYWORD")",
            // I will implement exactly that.

            std::vector<int> targets;
            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
            } else {
                 // Filter logic needed
                 // Can reuse TargetUtils logic or delegate
                 // Since I don't have full TargetUtils logic here to scan all zones easily without copy-paste,
                 // I will assume for now targets are provided or implicit logic matches `resolve` (which was buggy/incomplete in old code anyway regarding ID targets).
                 // But wait, old code just added a PassiveEffect globally!
                 // It did NOT iterate targets.
                 // So "Grant Keyword" was implemented as a GLOBAL RULE modification (Passive).

                 // If the user wants "InstructionOp::MODIFY", they assume it becomes an instance-based modification.
                 // This changes semantics: Passive applies to new cards entering too. Modify applies only to current.
                 // "All your creatures have speed attacker" (Passive) vs "All your creatures gain speed attacker" (One-shot Modify).

                 // I should support both?
                 // If targets are provided, use MODIFY.
                 // If not, maybe fallback to Passive? But Pipeline doesn't support ADD_PASSIVE.

                 // Let's implement MODIFY for now as requested.
                 // If no targets, we must find them.
                 // Copy logic from "TapHandler" to find implicit targets.

                 PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                 std::vector<std::pair<PlayerID, Zone>> zones_to_check;
                 // Default to Battle Zone
                 zones_to_check.push_back({controller_id, Zone::BATTLE});
                 if (ctx.action.filter.owner == "OPPONENT") zones_to_check.push_back({1 - controller_id, Zone::BATTLE});
                 else if (ctx.action.filter.owner == "BOTH") zones_to_check.push_back({1 - controller_id, Zone::BATTLE});

                 // Iterate and collect
                 for (const auto& [pid, zone] : zones_to_check) {
                    Player& p = ctx.game_state.players[pid];
                    for (auto& card : p.battle_zone) {
                         if (!ctx.card_db.count(card.card_id)) continue;
                         const auto& def = ctx.card_db.at(card.card_id);
                         if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                              targets.push_back(card.instance_id);
                         }
                    }
                }
            }

            if (targets.empty()) return;

            for (int t : targets) {
                 nlohmann::json args;
                 args["type"] = "ADD_KEYWORD";
                 args["str_value"] = ctx.action.str_val;
                 args["target"] = t;
                 // Duration? Not supported in MODIFY instruction args for ADD_KEYWORD yet.
                 // Assuming permanent or managed externally for now.

                 ctx.instruction_buffer->emplace_back(InstructionOp::MODIFY, args);
            }
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

        void resolve_with_targets(const ResolutionContext& ctx) override {
            resolve(ctx);
        }
    };
}
