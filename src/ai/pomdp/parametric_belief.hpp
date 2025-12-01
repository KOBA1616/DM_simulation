// Parametric belief: simple per-card probability distribution (header-only)
#pragma once

#include <map>
#include <vector>
#include <cstdint>
#include <numeric>
#include <algorithm>

#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"

namespace dm {
namespace ai {

using namespace dm::core;

class ParametricBelief {
public:
    ParametricBelief() {}

    // Initialize uniform prior over known card ids
    void initialize(const std::map<uint16_t, CardDefinition>& card_db) {
        probs.clear();
        if (card_db.empty()) return;
        float u = 1.0f / static_cast<float>(card_db.size());
        for (const auto &p : card_db) probs[p.first] = u;
    }

    // Initialize from a list of card ids (convenience for Python tests)
    void initialize_ids(const std::vector<uint16_t>& ids) {
        probs.clear();
        if (ids.empty()) return;
        float u = 1.0f / static_cast<float>(ids.size());
        for (auto id : ids) probs[id] = u;
    }

    // Naive update: reduce probability for cards that are visible in the provided state
    void update(const GameState &state) {
        if (probs.empty()) return;
        // Collect visible card ids from both players (hand, battle, shield, graveyard)
        std::map<uint16_t, int> seen_count;
        for (const auto &player : state.players) {
            for (const auto &c : player.hand) seen_count[c.card_id]++;
            for (const auto &c : player.battle_zone) seen_count[c.card_id]++;
            for (const auto &c : player.shield_zone) seen_count[c.card_id]++;
            for (const auto &c : player.graveyard) seen_count[c.card_id]++;
            for (const auto &c : player.deck) seen_count[c.card_id]++;
        }

        // Penalize probabilities for seen cards and renormalize
        for (auto &p : probs) {
            auto it = seen_count.find(p.first);
            if (it != seen_count.end()) {
                p.second *= 0.25f; // reduce confidence for observed ones
            }
        }

        // Renormalize to sum to 1 if possible
        float sum = 0.0f;
        for (const auto &p : probs) sum += p.second;
        if (sum <= 0.0f) return;
        for (auto &p : probs) p.second /= sum;
    }

    // Return belief vector as probabilities in ascending card id order
    std::vector<float> get_vector() const {
        std::vector<float> out;
        out.reserve(probs.size());
        for (const auto &p : probs) out.push_back(p.second);
        return out;
    }

private:
    std::map<uint16_t, float> probs;
};

} // namespace ai
} // namespace dm
