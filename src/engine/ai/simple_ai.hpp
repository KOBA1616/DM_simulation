#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include <vector>
#include <optional>

namespace dm::engine::ai {

/**
 * Simple priority-based AI for action selection
 * 
 * This AI selects actions based on fixed priorities:
 * 1. RESOLVE_EFFECT (must complete pending effects)
 * 2. PLAY_FROM_ZONE (play cards from hand)
 * 3. ATTACK (attack creatures/player)
 * 4. MANA_CHARGE (in MANA phase)
 * 5. Other actions
 * 6. PASS (exit phase)
 */
class SimpleAI {
public:
    /**
     * Select an action from available actions based on priority.
     * 
     * @param actions Vector of available actions
     * @param state Current game state (for future heuristics)
     * @return Index of selected action, or nullopt if no action available
     */
    static std::optional<size_t> select_action(
        const std::vector<core::CommandDef>& actions,
        const core::GameState& state
    );

private:
    /**
     * Get priority value for an action type based on current phase.
     * Higher value = higher priority.
     * 
     * @param action Action to evaluate
     * @param state Current game state (used for phase-aware priorities)
     * @return Priority value (0-100)
     */
    static int get_priority(const core::CommandDef& action, const core::GameState& state);
};

} // namespace dm::engine::ai
