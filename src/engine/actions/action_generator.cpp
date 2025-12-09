#include "action_generator.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> ActionGenerator::generate_legal_actions(const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        // Prepare Context
        ActionGenContext ctx = { game_state, card_db, game_state.active_player_id };

        // 1. Pending Effects
        if (!game_state.pending_effects.empty()) {
            PendingEffectStrategy pending_strategy;
            return pending_strategy.generate(ctx);
        }

        // 2. Stack (Atomic Action Flow)
        if (!game_state.stack_zone.empty()) {
            StackStrategy stack_strategy;
            return stack_strategy.generate(ctx);
        }

        // 3. Phase Specific Strategies
        if (game_state.current_phase == Phase::BLOCK) {
            BlockPhaseStrategy block_strategy;
            return block_strategy.generate(ctx);
        }

        switch (game_state.current_phase) {
            case Phase::START_OF_TURN:
            case Phase::DRAW:
                {
                    // No actions logic required for these phases in current design, just PASS
                    std::vector<Action> actions;
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                    return actions;
                }
            case Phase::MANA:
                {
                    ManaPhaseStrategy mana_strategy;
                    return mana_strategy.generate(ctx);
                }
            case Phase::MAIN:
                {
                    MainPhaseStrategy main_strategy;
                    return main_strategy.generate(ctx);
                }
            case Phase::ATTACK:
                {
                    AttackPhaseStrategy attack_strategy;
                    return attack_strategy.generate(ctx);
                }
            default:
                return {};
        }
    }

}
