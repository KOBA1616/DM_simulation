#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/game_command/commands.hpp"

namespace dm::engine {

    class PlayHandler : public IActionHandler {
    public:
        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Gather targets explicitly
            std::vector<int> targets;
            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
            } else if (!ctx.action.filter.zones.empty()) {
                // If no targets provided, find them based on filter (e.g. self)
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

            if (ctx.instruction_buffer) {
                for (int t_id : targets) {
                    // Generate Instructions
                    // 1. Move to Stack (using MOVE op)
                    // Note: We need to specify "to": "STACK" which requires Pipeline support.
                    // Assuming we updated PipelineExecutor to map "STACK" -> Zone::STACK.
                    nlohmann::json move_args;
                    move_args["target"] = t_id;
                    move_args["to"] = "STACK";
                    ctx.instruction_buffer->emplace_back(InstructionOp::MOVE, move_args);

                    // 2. Resolve Play
                    nlohmann::json play_args;
                    play_args["type"] = "RESOLVE_PLAY";
                    play_args["target"] = t_id;
                    play_args["value"] = ctx.action.value1; // cost reduction
                    play_args["spawn_source"] = static_cast<int>(SpawnSource::EFFECT_SUMMON);
                    ctx.instruction_buffer->emplace_back(InstructionOp::GAME_ACTION, play_args);
                }
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            // Deprecated: Just compile and run (Requires Pipeline)
            // For now, we keep legacy implementation or route to compile?
            // The prompt asks to migrate resolve logic to compile.
            // If we are fully migrating, resolve() should call pipeline.
            // But IActionHandler::resolve doesn't have access to pipeline executor easily without passing it.
            // Assuming this is called via PipelineExecutor in the future.

            // For now, retaining legacy logic to avoid breakage until full switch
            // OR implementing the "pipeline wrapper" if possible.
            // Since I cannot change the signature of resolve() to accept pipeline,
            // and creating a new pipeline here might be recursive.
            // I will implement legacy logic by calling `resolve_with_targets`.

            // Re-use legacy logic for now to pass tests while providing `compile`.
            using namespace dm::core;

            // Gather targets (implicit from filter)
            std::vector<int> targets;
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

            ResolutionContext sub_ctx = ctx;
            sub_ctx.targets = &targets;
            resolve_with_targets(sub_ctx);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            using namespace dm::engine::systems;

            if (!ctx.targets || ctx.targets->empty()) return;

            for (int target_id : *ctx.targets) {
                // Legacy imperative logic
                CardInstance* card = ctx.game_state.get_card_instance(target_id);
                if (!card) continue;

                PlayerID controller = ctx.game_state.active_player_id;
                if (ctx.source_instance_id >= 0 && ctx.source_instance_id < (int)ctx.game_state.card_owner_map.size()) {
                     controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
                }

                std::optional<CardInstance> removed_card = ZoneUtils::find_and_remove(ctx.game_state, target_id);
                if (!removed_card) continue;

                ctx.game_state.stack_zone.push_back(*removed_card);

                GameLogicSystem::resolve_play_from_stack(
                    ctx.game_state,
                    target_id,
                    ctx.action.value1,
                    SpawnSource::EFFECT_SUMMON,
                    controller,
                    ctx.card_db,
                    -1,
                    0
                );
            }
        }
    };
}
