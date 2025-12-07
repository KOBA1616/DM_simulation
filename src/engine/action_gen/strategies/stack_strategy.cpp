#include "stack_strategy.hpp"
#include "../../card_system/target_utils.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> StackStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& game_state = ctx.game_state;

        // 0.5. Check for Cards on Stack (Atomic Action Flow)
        if (!game_state.stack_zone.empty()) {
            const auto& stack_card = game_state.stack_zone.back();

            if (stack_card.is_tapped) {
                 // Cost is paid. Generate RESOLVE_PLAY.
                 Action resolve;
                 resolve.type = ActionType::RESOLVE_PLAY;
                 resolve.card_id = stack_card.card_id;
                 resolve.source_instance_id = stack_card.instance_id;
                 actions.push_back(resolve);
            } else {
                 // Cost not paid. Generate PAY_COST.
                 Action pay;
                 pay.type = ActionType::PAY_COST;
                 pay.card_id = stack_card.card_id;
                 pay.source_instance_id = stack_card.instance_id;
                 actions.push_back(pay);
            }
        }
        return actions;
    }

}
