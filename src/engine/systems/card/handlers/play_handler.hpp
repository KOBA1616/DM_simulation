#pragma once
#include "engine/systems/effects/effect_system.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/director/game_logic_system.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"

namespace dm::engine {

    class PlayHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
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
                     controller = ctx.game_state.get_card_owner(ctx.source_instance_id);
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
                                 if (dm::engine::utils::TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller, pid, false)) {
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
                // Determine if Evolution
                bool is_evolution = false;
                CardInstance* card = ctx.game_state.get_card_instance(target_id);
                if (card && ctx.card_db.count(card->card_id)) {
                    const auto& def = ctx.card_db.at(card->card_id);
                    if (def.keywords.evolution) is_evolution = true;
                }

                if (is_evolution) {
                    // Use PLAY instruction to handle Evolution logic (Select Base -> Attach)
                    nlohmann::json play_args;
                    play_args["card"] = target_id;
                    ctx.instruction_buffer->emplace_back(InstructionOp::PLAY, play_args);
                } else {
                    // Simple Move to Battle Zone (or Stack for Spells, though PlayHandler usually implies Battle)
                    // PipelineExecutor now supports "STACK".
                    std::string dest = "BATTLE";
                    if (card && ctx.card_db.count(card->card_id)) {
                        const auto& def = ctx.card_db.at(card->card_id);
                        if (def.type == CardType::SPELL) dest = "STACK";
                    }

                    nlohmann::json move_args;
                    move_args["target"] = target_id;
                    move_args["to"] = dest;
                    ctx.instruction_buffer->emplace_back(InstructionOp::MOVE, move_args);
                }
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            // Backward compatibility wrapper
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
