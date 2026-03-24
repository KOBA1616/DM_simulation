#pragma once

#include "core/game_state.hpp"

namespace dm::engine::game_command {

// Centralized stat-update helpers for turn-scoped counters.
// Use these functions to ensure replacement effects are respected
// and to provide a single extension point for logging/telemetry.
void add_turn_destroyed_count(core::GameState &state, int amount);

} // namespace dm::engine::game_command
