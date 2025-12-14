#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include <algorithm>

namespace dm::engine {

    class MoveToUnderCardHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            // Source is usually implicit (e.g. top of deck) or defined by source_zone.
            // But MOVE_TO_UNDER_CARD implies moving TO a target card.
            // resolve() without targets is ambiguous unless there is a single valid target?
            // Usually this action is targeted.
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;

             // Logic: Move card FROM source (e.g. top of deck, hand) TO under the target card.
             // source_zone defined in action.

             PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
             Player& controller = ctx.game_state.players[controller_id];

             std::vector<CardInstance> sources;

             // Get Source Card(s)
             if (ctx.action.source_zone == "DECK" || ctx.action.source_zone.empty()) {
                 if (!controller.deck.empty()) {
                     sources.push_back(controller.deck.back());
                     controller.deck.pop_back();
                 }
             } else if (ctx.action.source_zone == "HAND") {
                 // Requires selection from hand? Or random? Or specific?
                 // Usually "Put a card from your hand under..." implies selection.
                 // This handler assumes the "Moving" part. Selection should be handled before?
                 // If action is "Move chosen card from hand", then `ctx.targets` contains the card from hand.
                 // But `ctx.targets` here usually refers to the DESTINATION card (the one getting the card under it).

                 // If the source is also selected, we need context.
                 // Complex hierarchy building (e.g. Phoenix) usually involves:
                 // 1. Select Card to Evolve (Target)
                 // 2. Move Source (Self) onto Target? No, that's Evolution.

                 // Case: "Put top card of deck under this creature" (e.g. Castro)
                 // Target: This creature. Source: Deck.
             }

             if (sources.empty()) return;

             // Move to Target(s)
             for (int tid : *ctx.targets) {
                 CardInstance* target_card = ctx.game_state.get_card_instance(tid);
                 if (target_card) {
                     for (const auto& s : sources) {
                         target_card->underlying_cards.push_back(s);
                     }
                 }
             }
        }
    };
}
