#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/effects/effect_resolver.hpp"
#include "engine/game_command/commands.hpp"

namespace dm::engine {

    class PlayHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Delegate if needed
            if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 // Usually PLAY_CARD doesn't use generic select logic directly but relies on "From Zone" implicit iteration
                 // unless explicitly set.
                 // However, existing logic supports it.
            }

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

            if (!ctx.targets || ctx.targets->empty()) return;

            for (int target_id : *ctx.targets) {
                CardInstance* card = ctx.game_state.get_card_instance(target_id);
                if (!card) continue;

                PlayerID controller = ctx.game_state.active_player_id;
                // Determine controller for the PLAY action
                if (ctx.source_instance_id >= 0 && ctx.source_instance_id < (int)ctx.game_state.card_owner_map.size()) {
                     controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
                }

                // Identify source zone for TransitionCommand
                // We need to know where it is coming from.
                Zone from_zone = Zone::HAND;
                PlayerID owner_id = 0;
                bool found = false;

                for (auto& p : ctx.game_state.players) {
                    if (std::any_of(p.hand.begin(), p.hand.end(), [&](const auto& c){ return c.instance_id == target_id; })) { from_zone = Zone::HAND; owner_id = p.id; found = true; break; }
                    if (std::any_of(p.graveyard.begin(), p.graveyard.end(), [&](const auto& c){ return c.instance_id == target_id; })) { from_zone = Zone::GRAVEYARD; owner_id = p.id; found = true; break; }
                    if (std::any_of(p.mana_zone.begin(), p.mana_zone.end(), [&](const auto& c){ return c.instance_id == target_id; })) { from_zone = Zone::MANA; owner_id = p.id; found = true; break; }
                    if (std::any_of(p.battle_zone.begin(), p.battle_zone.end(), [&](const auto& c){ return c.instance_id == target_id; })) { from_zone = Zone::BATTLE; owner_id = p.id; found = true; break; }
                    if (std::any_of(p.shield_zone.begin(), p.shield_zone.end(), [&](const auto& c){ return c.instance_id == target_id; })) { from_zone = Zone::SHIELD; owner_id = p.id; found = true; break; }
                    if (std::any_of(p.deck.begin(), p.deck.end(), [&](const auto& c){ return c.instance_id == target_id; })) { from_zone = Zone::DECK; owner_id = p.id; found = true; break; }
                }

                if (!found) continue;

                // 1. Move to Stack (using TransitionCommand)
                // TransitionCommand supports Zone::STACK via its internal logic
                TransitionCommand cmd(target_id, from_zone, Zone::STACK, owner_id);
                cmd.execute(ctx.game_state);

                // 2. Play Resolution
                // The card is now in the Stack. Resolve it.
                int cost_reduction = ctx.action.value1;

                EffectResolver::resolve_play_from_stack(
                    ctx.game_state,
                    target_id,
                    cost_reduction,
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
