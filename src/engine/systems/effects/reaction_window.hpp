#pragma once
#include "core/types.hpp"
#include <vector>

namespace dm::engine::systems {

    enum class ReactionType {
        SHIELD_TRIGGER,
        REVOLUTION_CHANGE,
        NONE
    };

    struct ReactionCandidate {
        int card_id;
        int instance_id;
        dm::core::PlayerID player_id;
        ReactionType type;
    };

    class ReactionWindow {
    public:
        std::vector<ReactionCandidate> candidates;
        std::vector<int> used_candidate_indices; // Track used options
        ReactionWindow(const std::vector<ReactionCandidate>& c) : candidates(c) {}
        ReactionWindow() = default;
    };
}
