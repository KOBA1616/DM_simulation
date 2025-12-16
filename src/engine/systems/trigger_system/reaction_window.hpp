#pragma once
#include "core/types.hpp"
#include <vector>

namespace dm::engine::systems {

    enum class ReactionType {
        SHIELD_TRIGGER,
        REVOLUTION_CHANGE,
        NINJA_STRIKE,
        STRIKE_BACK
    };

    struct ReactionCandidate {
        dm::core::CardID card_id;
        int instance_id;
        dm::core::PlayerID player_id;
        ReactionType type;
    };

    struct ReactionWindow {
        std::vector<ReactionCandidate> candidates;

        // Tracks which candidates have been processed or chosen.
        // For S-Trigger, you can use multiple.
        // For Ninja Strike, you can use multiple (but sequential).
        // For Revolution Change, usually one per attack (rule variation exists, but generally one change per creature).
        std::vector<int> used_candidate_indices;

        // If true, the window closes only after all players pass.
        bool active = true;

        ReactionWindow() = default;
        ReactionWindow(const std::vector<ReactionCandidate>& c) : candidates(c) {}
    };

}
