#include "action_generator.hpp"
#include "core/constants.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> ActionGenerator::generate_legal_actions(const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        // Phase 6: Handle Waiting for User Input (Query Response)
        if (game_state.waiting_for_user_input && game_state.pending_query) {
            std::vector<Action> actions;
            const auto& query = *game_state.pending_query;

            if (query.query_type == "SELECT_TARGET") {
                for (int target_id : query.valid_target_ids) {
                    Action act;
                    act.type = ActionType::SELECT_TARGET;
                    act.target_instance_id = target_id;
                    // We might need to encode query_id to ensure we are answering the right query,
                    // but Action struct is limited. The engine state implies the context.
                    actions.push_back(act);
                }
            }
            else if (query.query_type == "SELECT_OPTION") {
                for (size_t i = 0; i < query.options.size(); ++i) {
                    Action act;
                    act.type = ActionType::SELECT_OPTION;
                    act.target_slot_index = static_cast<int>(i);
                    // Optionally store string value in a future Action extension
                    actions.push_back(act);
                }
            }

            // If no valid choices (shouldn't happen for well-formed queries), return empty or pass?
            // Usually queries have at least one option.
            return actions;
        }

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
