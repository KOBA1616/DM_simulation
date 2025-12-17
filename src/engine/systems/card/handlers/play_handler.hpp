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
        void resolve(const ResolutionContext& ctx) override {
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
            using namespace dm::engine::game_command;
            using namespace dm::engine::systems;

            if (!ctx.targets || ctx.targets->empty()) return;

            for (int target_id : *ctx.targets) {
                // 1. Remove from current zone (Manual, as TransitionCommand lacks Stack)
                std::optional<CardInstance> removed_card = ZoneUtils::find_and_remove(ctx.game_state, target_id);
                if (!removed_card) continue;

                // 2. Add to Stack
                ctx.game_state.stack_zone.push_back(*removed_card);

                // 3. Play via Pipeline/GameLogicSystem
                // We construct an Instruction to call handle_resolve_play
                nlohmann::json args;
                args["type"] = "RESOLVE_PLAY";
                args["source_id"] = target_id;
                // Inherit cost reduction if any (value1)
                // Actually RESOLVE_PLAY in GameLogicSystem doesn't take reduction, it assumes paid or free.
                // This is effect summon, so it's free.
                args["spawn_source"] = (int)SpawnSource::EFFECT_SUMMON;
                // dest_override?

                // Since IActionHandler is synchronous, we can just call GameLogicSystem::resolve_action_oneshot
                // Or better, use a temporary Pipeline.

                // Construct Action to pass to GameLogicSystem
                Action resolve_act;
                resolve_act.type = ActionType::RESOLVE_PLAY;
                resolve_act.source_instance_id = target_id;
                resolve_act.spawn_source = SpawnSource::EFFECT_SUMMON;

                GameLogicSystem::resolve_action_oneshot(ctx.game_state, resolve_act, ctx.card_db);
            }
        }
    };
}
