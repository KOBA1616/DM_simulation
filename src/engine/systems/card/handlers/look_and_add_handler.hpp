#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include <algorithm>

namespace dm::engine {

    class LookAndAddHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
            Player& controller = ctx.game_state.players[controller_id];

            int look_count = ctx.action.value1;
            int add_count = ctx.action.value2;

            if (look_count <= 0) return;
            if (add_count <= 0) add_count = 1;

            std::vector<CardInstance> looked_cards;
            for (int i = 0; i < look_count; ++i) {
                if (controller.deck.empty()) break;
                looked_cards.push_back(controller.deck.back());
                controller.deck.pop_back();
            }

            if (looked_cards.empty()) return;

            // Move looked cards to effect buffer for selection
            controller.effect_buffer.insert(controller.effect_buffer.end(), looked_cards.begin(), looked_cards.end());

            // Create a pending effect to select 'add_count' cards
            PendingEffect pending(EffectType::NONE, ctx.source_instance_id, controller_id);
            pending.resolve_type = ResolveType::TARGET_SELECT;

            pending.filter = ctx.action.filter;
            pending.filter.zones = {"EFFECT_BUFFER"};

            pending.num_targets_needed = add_count;

            // Continuation Effect
            EffectDef continuation;

            // Action A: Move selected cards to hand
            ActionDef act_move;
            act_move.type = EffectActionType::MOVE_CARD;
            act_move.source_zone = "EFFECT_BUFFER";
            act_move.destination_zone = "HAND";
            continuation.actions.push_back(act_move);

            // Action B: Move remaining to bottom
            ActionDef act_cleanup;
            act_cleanup.type = EffectActionType::MOVE_BUFFER_TO_ZONE;
            act_cleanup.destination_zone = "DECK_BOTTOM";
            continuation.actions.push_back(act_cleanup);

            pending.effect_def = continuation;

            ctx.game_state.pending_effects.push_back(pending);
        }

        void resolve_with_targets(const ResolutionContext& /*ctx*/) override {
        }
    };
}
