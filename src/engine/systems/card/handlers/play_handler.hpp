#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/effects/effect_resolver.hpp"

namespace dm::engine {

    class PlayHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            // Usually requires targets, but if "filter" is set and "scope" is not TARGET_SELECT,
            // GenericCardSystem iterates. However, PLAY_FROM_ZONE usually uses TARGET_SELECT logic
            // or iterates zones.

            using namespace dm::core;

            // Gather targets from source zone(s) based on filter
            // Note: action.source_zone is deprecated/legacy, we rely on action.filter.zones.

            std::vector<int> targets;

            if (ctx.action.filter.zones.empty()) {
                return;
            }

            // Find all valid targets
            PlayerID controller = ctx.game_state.active_player_id; // Or owner of effect?
            // The context has source_instance_id.

            // We need to fetch the CardDefinition for the source instance to check ownership if needed
            // But we can just use card_owner_map

            if (ctx.source_instance_id >= 0 && ctx.source_instance_id < (int)ctx.game_state.card_owner_map.size()) {
                 controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
            }

            // Which player's zones?
            // action.filter.owner
            std::vector<PlayerID> target_players;
            if (ctx.action.filter.owner == "OPPONENT") {
                target_players.push_back(1 - controller);
            } else if (ctx.action.filter.owner == "BOTH") {
                target_players.push_back(controller);
                target_players.push_back(1 - controller);
            } else {
                target_players.push_back(controller);
            }

            for (PlayerID pid : target_players) {
                // For each zone in filter
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
                        // Check if card definition exists
                        if (ctx.card_db.count(card.card_id)) {
                             const auto& def = ctx.card_db.at(card.card_id);
                             if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller, pid, false)) {
                                 targets.push_back(card.instance_id);
                             }
                        }
                    }
                }
            }

            // Delegate to resolve_with_targets
            ResolutionContext sub_ctx = ctx;
            sub_ctx.targets = &targets;
            resolve_with_targets(sub_ctx);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (!ctx.targets || ctx.targets->empty()) return;

            for (int target_id : *ctx.targets) {
                CardInstance* card = ctx.game_state.get_card_instance(target_id);
                if (!card) continue;

                PlayerID controller = ctx.game_state.active_player_id;

                if (ctx.source_instance_id >= 0 && ctx.source_instance_id < (int)ctx.game_state.card_owner_map.size()) {
                     controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
                }

                // 1. Remove from current zone
                std::optional<CardInstance> removed_card = ZoneUtils::find_and_remove(ctx.game_state, target_id);
                if (!removed_card) continue;

                // 2. Add to Stack
                ctx.game_state.stack_zone.push_back(*removed_card);

                // 3. Play
                int cost_reduction = ctx.action.value1;

                EffectResolver::resolve_play_from_stack(
                    ctx.game_state,
                    target_id,
                    cost_reduction,
                    SpawnSource::EFFECT_SUMMON, // Generalize as EFFECT
                    controller,
                    ctx.card_db,
                    -1, // evo_source_id
                    0   // dest_override
                );
            }
        }
    };
}
