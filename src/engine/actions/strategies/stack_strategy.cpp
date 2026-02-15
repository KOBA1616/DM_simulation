#include "stack_strategy.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/systems/mechanics/mana_system.hpp"
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    std::vector<CommandDef> StackStrategy::generate(const ActionGenContext& ctx) {
        std::vector<CommandDef> actions;
        const auto& game_state = ctx.game_state;
        const auto& card_db = ctx.card_db;
        const Player& active_player = game_state.players[game_state.active_player_id];

        // 0.5. Check for Cards on Stack (Atomic Action Flow)
        // Check pending effects (legacy/stack effects)
        if (!game_state.pending_effects.empty()) {
            const auto& pending = game_state.pending_effects.back();
            if (pending.type == EffectType::INTERNAL_PLAY) {
                 CommandDef resolve;
                 resolve.type = CommandType::RESOLVE_PLAY;
                 resolve.instance_id = pending.source_instance_id;
                 actions.push_back(resolve);
            }
        }

        // Check active player's stack zone
        for (const auto& card : active_player.stack) {
            if (card.is_tapped) {
                // Already paid -> RESOLVE_PLAY
                CommandDef resolve;
                resolve.type = CommandType::RESOLVE_PLAY;
                resolve.instance_id = card.instance_id;
                actions.push_back(resolve);
            } else {
                // Not paid -> PLAY_FROM_ZONE (Triggers Pay + Resolve)
                // Check if cost CAN be paid before generating the action
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    // Check if player can pay the cost
                    if (ManaSystem::can_pay_cost(game_state, active_player, def, card_db)) {
                        CommandDef pay;
                        pay.type = CommandType::PLAY_FROM_ZONE;
                        pay.instance_id = card.instance_id;
                        actions.push_back(pay);
                    }
                }
            }
        }

        return actions;
    }

}
