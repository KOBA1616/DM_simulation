#include "stack_strategy.hpp"
#include "engine/systems/card/target_utils.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> StackStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& game_state = ctx.game_state;

        // 0.5. Check for Cards on Stack (Atomic Action Flow)
        if (!game_state.pending_effects.empty()) {
            const auto& pending = game_state.pending_effects.back();
            // This logic seems to assume we are tracking payment state in the pending effect.
            // Since PendingEffect doesn't track "is_paid" explicitly in the same way CardInstance.is_tapped might,
            // we need a robust way.
            // For now, let's assume if it's INTERNAL_PLAY, we need to resolve it.

            // Note: The original logic accessed stack_card.is_tapped, which was likely abuse of the field
            // to store "paid" status in a legacy implementation.

            if (pending.type == EffectType::INTERNAL_PLAY) {
                 // Assuming always resolved if in this state for now, or check external state.
                 // Actually, PAY_COST actions usually happen *before* putting on stack or via specific prompts.
                 // If it's on pending_effects, it's ready to resolve.

                 Action resolve;
                 resolve.type = ActionType::RESOLVE_PLAY;
                 resolve.source_instance_id = pending.source_instance_id;
                 actions.push_back(resolve);
            }
        }
        return actions;
    }

}
