#include "stack_strategy.hpp"
#include "engine/systems/card/target_utils.hpp"
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> StackStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& game_state = ctx.game_state;
        const auto& card_db = ctx.card_db;
        const Player& active_player = game_state.players[game_state.active_player_id];

        // 0.5. Check for Cards on Stack (Atomic Action Flow)
        // Check pending effects (legacy/stack effects)
        if (!game_state.pending_effects.empty()) {
            const auto& pending = game_state.pending_effects.back();
            if (pending.type == EffectType::INTERNAL_PLAY) {
                 Action resolve;
                 resolve.type = PlayerIntent::RESOLVE_PLAY;
                 resolve.source_instance_id = pending.source_instance_id;
                 actions.push_back(resolve);
            }
        }

        // Check active player's stack zone
        for (const auto& card : active_player.stack) {
            if (card.is_tapped) {
                // Already paid -> RESOLVE_PLAY
                Action resolve;
                resolve.type = PlayerIntent::RESOLVE_PLAY;
                resolve.source_instance_id = card.instance_id;
                resolve.card_id = card.card_id;
                actions.push_back(resolve);
            } else {
                // Not paid -> PAY_COST
                // Validate if cost CAN be paid?
                // For generation, we usually assume it's legal to TRY to pay.
                // Or check ManaSystem::can_pay_cost?
                // Since DECLARE_PLAY already happened (which usually checks cost legality),
                // we assume payment is possible or at least attempted.
                Action pay;
                pay.type = PlayerIntent::PAY_COST;
                pay.source_instance_id = card.instance_id;
                pay.card_id = card.card_id;
                actions.push_back(pay);
            }
        }

        return actions;
    }

}
