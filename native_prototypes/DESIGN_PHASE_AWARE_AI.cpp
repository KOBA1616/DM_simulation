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

    // Find action with highest priority in current phase
    size_t best_idx = 0;
    int best_priority = get_priority(actions[0], state);

    for (size_t i = 1; i < actions.size(); ++i) {
        int priority = get_priority(actions[i], state);
        if (priority > best_priority) {
            best_priority = priority;
            best_idx = i;
        }
    }

    std::cout << "[SimpleAI] Phase=" << static_cast<int>(state.current_phase)
              << " Selected action #" << best_idx 
              << " with priority " << best_priority 
              << " (type=" << static_cast<int>(actions[best_idx].type) << ")\n";

    return best_idx;
}

int SimpleAI::get_priority(const Action& action, const GameState& state) {
    // Universal highest priority: must resolve pending effects
    if (action.type == PlayerIntent::RESOLVE_EFFECT) {
        return 100;
    }
    
    // Query responses (target selection, options) are always high priority
    if (action.type == PlayerIntent::SELECT_TARGET || 
        action.type == PlayerIntent::SELECT_OPTION) {
        return 90;
    }
    
    // PASS is always lowest priority
    if (action.type == PlayerIntent::PASS) {
        return 0;
    }
    
    // Phase-specific priorities
    switch (state.current_phase) {
        case GamePhase::MANA:
            // In MANA phase, charging mana is the primary action
            if (action.type == PlayerIntent::MANA_CHARGE) {
                return 90;  // Very high priority
            }
            return 10;  // Other actions are low priority
        
        case GamePhase::MAIN:
            // In MAIN phase, prioritize playing cards and attacking
            if (action.type == PlayerIntent::PLAY_CARD || 
                action.type == PlayerIntent::PLAY_CARD_INTERNAL) {
                return 80;  // High priority
            }
            if (action.type == PlayerIntent::ATTACK_PLAYER || 
                action.type == PlayerIntent::ATTACK_CREATURE) {
                return 60;  // Medium-high priority
            }
            if (action.type == PlayerIntent::MANA_CHARGE) {
                return 10;  // Low priority (wrong phase)
            }
            return 20;  // Other actions (abilities, etc.)
        
        case GamePhase::ATTACK_DECLARE:
            // In ATTACK_DECLARE phase, declaring attacks is primary
            if (action.type == PlayerIntent::ATTACK_PLAYER || 
                action.type == PlayerIntent::ATTACK_CREATURE) {
                return 85;  // Very high priority
            }
            return 10;  // Other actions are low priority
        
        case GamePhase::BLOCK_DECLARE:
            // In BLOCK_DECLARE phase, declaring blockers is primary
            if (action.type == PlayerIntent::DECLARE_BLOCKER) {
                return 85;  // Very high priority
            }
            if (action.type == PlayerIntent::DECLARE_NO_BLOCK) {
                return 10;  // Low priority (prefer blocking if possible)
            }
            return 10;  // Other actions are low priority
        
        default:
            // For other phases (START, DRAW, END, DAMAGE, etc.)
            // Use conservative fixed priorities
            switch (action.type) {
                case PlayerIntent::PLAY_CARD:
                case PlayerIntent::PLAY_CARD_INTERNAL:
                    return 60;
                
                case PlayerIntent::ATTACK_PLAYER:
                case PlayerIntent::ATTACK_CREATURE:
                    return 50;
                
                case PlayerIntent::MANA_CHARGE:
                    return 40;
                
                case PlayerIntent::DECLARE_BLOCKER:
                case PlayerIntent::DECLARE_NO_BLOCK:
                    return 50;
                
                default:
                    return 20;  // Other actions
            }
    }
}

} // namespace dm::engine::ai
