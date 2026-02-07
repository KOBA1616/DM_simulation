#include "simple_ai.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine::ai {

using namespace dm::core;

std::optional<size_t> SimpleAI::select_action(
    const std::vector<Action>& actions,
    const GameState& state
) {
    if (actions.empty()) {
        return std::nullopt;
    }

    // Find action with highest priority
    size_t best_idx = 0;
    int best_priority = get_priority(actions[0], state);

    for (size_t i = 1; i < actions.size(); ++i) {
        int priority = get_priority(actions[i], state);
        if (priority > best_priority) {
            best_priority = priority;
            best_idx = i;
        }
    }

    std::cout << "[SimpleAI] Selected action #" << best_idx 
              << " with priority " << best_priority 
              << " (type=" << static_cast<int>(actions[best_idx].type) << ")\n";

    return best_idx;
}

int SimpleAI::get_priority(const Action& action, const GameState& state) {
    // Universal priorities (phase-independent)
    if (action.type == PlayerIntent::RESOLVE_EFFECT) return 100;
    
    // Query responses - always high priority
    if (action.type == PlayerIntent::SELECT_TARGET ||
        action.type == PlayerIntent::SELECT_OPTION ||
        action.type == PlayerIntent::SELECT_NUMBER) return 95;
    
    // Stack actions - very high priority
    if (action.type == PlayerIntent::PAY_COST ||
        action.type == PlayerIntent::RESOLVE_PLAY) return 98;
    
    // PASS - always lowest priority
    if (action.type == PlayerIntent::PASS) return 0;
    
    // Phase-specific priorities
    switch (state.current_phase) {
        case Phase::MANA:
            // In MANA phase, prioritize MANA_CHARGE
            if (action.type == PlayerIntent::MANA_CHARGE) return 90;
            return 10;  // Other actions have low priority
        
        case Phase::MAIN:
            // In MAIN phase, prioritize card plays and attacks
            if (action.type == PlayerIntent::DECLARE_PLAY) return 80;
            if (action.type == PlayerIntent::PLAY_CARD ||
                action.type == PlayerIntent::PLAY_CARD_INTERNAL) return 80;
            if (action.type == PlayerIntent::ATTACK_PLAYER ||
                action.type == PlayerIntent::ATTACK_CREATURE) return 60;
            if (action.type == PlayerIntent::USE_ABILITY) return 50;
            if (action.type == PlayerIntent::MANA_CHARGE) return 10;  // Wrong phase
            return 20;
        
        case Phase::ATTACK:
            // In ATTACK phase, prioritize attacks
            if (action.type == PlayerIntent::ATTACK_PLAYER ||
                action.type == PlayerIntent::ATTACK_CREATURE) return 85;
            return 10;
        
        case Phase::BLOCK:
            // In BLOCK phase, prioritize blocking
            if (action.type == PlayerIntent::BLOCK) return 85;
            return 10;
        
        default:
            // For other phases or general actions
            if (action.type == PlayerIntent::PLAY_CARD ||
                action.type == PlayerIntent::PLAY_CARD_INTERNAL) return 50;
            return 20;
    }
}

} // namespace dm::engine::ai
