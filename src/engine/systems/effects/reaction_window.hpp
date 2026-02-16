#pragma once
#include "core/types.hpp"
#include <vector>

namespace dm::engine::systems {

    enum class ReactionType {
        SHIELD_TRIGGER,
        REVOLUTION_CHANGE,
        NINJA_STRIKE,
        STRIKE_BACK,
        OTHER
    };

    struct ReactionCandidate {
        dm::core::CardID card_id;
        int instance_id;
        dm::core::PlayerID player_id;
        ReactionType type;
    };

    struct ReactionWindow {
        std::vector<ReactionCandidate> candidates;
        std::vector<int> used_candidate_indices;

        ReactionWindow(const std::vector<ReactionCandidate>& c) : candidates(c) {}
        ReactionWindow() = default;
    };

}
