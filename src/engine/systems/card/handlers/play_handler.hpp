#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "engine/game_command/commands.hpp"

namespace dm::engine {

    class PlayHandler : public IActionHandler {
    public:
        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Gather targets (implicit from filter if target_choice is not SELECT)
            std::vector<int> targets;
            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
            } else if (ctx.action.target_choice != "SELECT") {
                // Auto-select targets based on filter logic if no explicit targets provided
                // Copy logic from old resolve() to find implicit targets
                if (ctx.action.filter.zones.empty()) return;

                PlayerID controller = ctx.game_state.active_player_id;
                if (ctx.source_instance_id >= 0 && ctx.source_instance_id < (int)ctx.game_state.card_owner_map.size()) {
                     controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
                }

                std::vector<PlayerID> target_players;
                if (ctx.action.filter.owner == "OPPONENT") target_players.push_back(1 - controller);
                else if (ctx.action.filter.owner == "BOTH") { target_players.push_back(controller); target_players.push_back(1 - controller); }
                else target_players.push_back(controller);

                for (PlayerID pid : target_players) {
                    for (const auto& zone_name : ctx.action.filter.zones) {
                        const std::vector<CardInstance>* zone = nullptr;
                        if (zone_name == "HAND") zone = &ctx.game_state.players[pid].hand;
                        else if (zone_name == "GRAVEYARD") zone = &ctx.game_state.players[pid].graveyard;
                        else if (zone_name == "MANA_ZONE") zone = &ctx.game_state.players[pid].mana_zone;
                        else if (zone_name == "BATTLE_ZONE") zone = &ctx.game_state.players[pid].battle_zone;
                        else if (zone_name == "SHIELD_ZONE") zone = &ctx.game_state.players[pid].shield_zone;
                        else if (zone_name == "DECK") zone = &ctx.game_state.players[pid].deck;

                        if (!zone) continue;

                        for (const auto& card : *zone) {
                            if (ctx.card_db.count(card.card_id)) {
                                 const auto& def = ctx.card_db.at(card.card_id);
                                 if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller, pid, false)) {
                                     targets.push_back(card.instance_id);
                                 }
                            }
                        }
                    }
                }
            }

            if (targets.empty()) return;

            // Generate instructions
            for (int target_id : targets) {
                // 1. Move to Stack (using MOVE instruction)
                // We assume source is wherever it currently is (HAND, GRAVEYARD, etc.)
                // The MOVE instruction's handler automatically finds the card.
                nlohmann::json move_args;
                move_args["target"] = target_id;
                move_args["to"] = "STACK"; // Zone::STACK logic handled by TransitionCommand via MOVE
                // PipelineExecutor::handle_move currently doesn't explicitly support "STACK" string mapping in provided code?
                // Checked PipelineExecutor.cpp:
                // if (to_zone_str == "HAND") ...
                // It does NOT support "STACK".
                // I need to update PipelineExecutor to support "STACK" or use GAME_ACTION PLAY_CARD logic.
                // However, GAME_ACTION PLAY_CARD expects "source_id" and assumes HAND in current GameLogicSystem::handle_play_card.
                //
                // Workaround: We can assume GameLogicSystem::handle_resolve_play handles STACK.
                // So we need to move it to stack.
                // Since PipelineExecutor doesn't map STACK, I should add it or use a raw GameCommand if possible?
                // No, I should fix PipelineExecutor or use GAME_ACTION that delegates to a command.

                // Let's use GAME_ACTION "PLAY_CARD_INTERNAL" which does the move?
                // GameLogicSystem::dispatch_action(PLAY_CARD_INTERNAL) -> handle_play_card_internal (not implemented separately but inline in switch)
                // Switch case for PLAY_CARD_INTERNAL:
                // "Need to handle moving from hand if necessary... state.stack_zone.push_back(c)"
                // It does minimal manual move.

                // Better approach: Since I am refactoring, I should add "STACK" support to PipelineExecutor::handle_move.
                // But I cannot modify PipelineExecutor in this step easily without verifying.
                // Wait, I can modify PipelineExecutor.cpp in the same plan.

                // For now, I will assume "STACK" is supported or I will add it.
                // Actually, let's use a workaround: GAME_ACTION "PLAY_CARD" with a special flag?
                // No, sticking to "STACK" zone support is cleaner.
                // I will update PipelineExecutor::handle_move in a separate step or assume I'll do it.
                //
                // Wait, for this specific file, I'll generate:
                // InstructionOp::GAME_ACTION { type: "PLAY_CARD_INTERNAL", source_id: target_id, spawn_source: EFFECT_SUMMON }
                // This seems to be supported by GameLogicSystem::dispatch_action.

                nlohmann::json play_args;
                play_args["type"] = "PLAY_CARD_INTERNAL";
                play_args["source_id"] = target_id;
                play_args["spawn_source"] = (int)dm::core::SpawnSource::EFFECT_SUMMON;
                play_args["dest_override"] = 0;
                play_args["target_player"] = ctx.game_state.active_player_id; // Controller

                ctx.instruction_buffer->emplace_back(InstructionOp::GAME_ACTION, play_args);
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            // Backward compatibility wrapper
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
