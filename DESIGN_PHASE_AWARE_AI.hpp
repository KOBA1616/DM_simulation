#pragma once
#include "core/game_state.hpp"
#include "core/action.hpp"
#include <vector>
#include <optional>

namespace dm::engine::ai {

/**
 * Simple priority-based AI for action selection
 * 
 * This AI selects actions based on phase-aware priorities:
 * - RESOLVE_EFFECT is always highest priority (must complete pending effects)
 * - Other actions have different priorities depending on current phase
 * 
 * Phase-specific priorities:
 * - MANA phase: Prioritize MANA_CHARGE
 * - MAIN phase: Prioritize PLAY_CARD, then ATTACK
 * - ATTACK_DECLARE phase: Prioritize ATTACK
 * - BLOCK_DECLARE phase: Prioritize DECLARE_BLOCKER
 */
class SimpleAI {
public:
    /**
     * Select an action from available actions based on phase-aware priority.
     * 
     * @param actions Vector of available actions
     * @param state Current game state (used for phase detection)
     * @return Index of selected action, or nullopt if no action available
     */
    static std::optional<size_t> select_action(
        const std::vector<core::Action>& actions,
        const core::GameState& state
    );

private:
    /**
     * Get priority score for an action in the current game phase.
     * Higher scores = higher priority.
     * 
     * @param action Action to evaluate
     * @param state Current game state
     * @return Priority score (0-100)
     */
    static int get_priority(const core::Action& action, const core::GameState& state);
};

} // namespace dm::engine::ai
