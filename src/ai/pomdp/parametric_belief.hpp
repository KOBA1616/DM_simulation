// Parametric belief: simple per-card probability distribution (header-only)
#pragma once

#include <map>
#include <vector>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <set>

#include "core/game_state.hpp"
#include "core/card_def.hpp"

namespace dm {
namespace ai {

using namespace dm::core;

class ParametricBelief {
public:
    ParametricBelief() {}

    // Configure weights: strong_weight applies to hand/battle/shield/graveyard
    // deck_weight applies to cards appearing in deck listings (weaker evidence)
    void set_weights(float strong_w, float deck_w) {
        strong_weight = strong_w;
        deck_weight = deck_w;
    }

    std::pair<float,float> get_weights() const {
        return {strong_weight, deck_weight};
    }

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
        // Collect visible card ids from both players and zone-weight them.
        // Cards seen in hand/battle/shield/graveyard are strong evidence and should be penalized more.
        // Cards appearing in deck listings are weaker evidence (penalize less).
        std::map<uint16_t, float> seen_weight;
        for (const auto &player : state.players) {
            for (const auto &c : player.hand) seen_weight[c.card_id] += strong_weight; // strong
            for (const auto &c : player.battle_zone) seen_weight[c.card_id] += strong_weight; // strong
            for (const auto &c : player.shield_zone) seen_weight[c.card_id] += strong_weight; // strong
            for (const auto &c : player.graveyard) seen_weight[c.card_id] += strong_weight; // strong
            for (const auto &c : player.deck) seen_weight[c.card_id] += deck_weight; // weak evidence
        }

        // Penalize probabilities for seen cards proportionally to weight, then renormalize
        for (auto &p : probs) {
            auto it = seen_weight.find(p.first);
            if (it != seen_weight.end() && it->second > 0.0f) {
                // stronger weight => stronger penalty; map weight to multiplier in (0,1]
                float weight = it->second;
                float multiplier = 1.0f / (1.0f + weight); // e.g., weight 1 -> 0.5, weight 4 -> 0.2
                p.second *= multiplier;
            }
        }

        // Renormalize to sum to 1 if possible
        float sum = 0.0f;
        for (const auto &p : probs) sum += p.second;
        if (sum <= 0.0f) return;
        for (auto &p : probs) p.second /= sum;
    }

    // Update with previous state: detect transitions (deck -> hand/battle/etc.)
    // and treat them as reveals (stronger evidence) by adding an extra reveal weight.
    void update_with_prev(const GameState &prev_state, const GameState &state) {
        if (probs.empty()) return;

        std::map<uint16_t, float> seen_weight;
        for (const auto &player : state.players) {
            for (const auto &c : player.hand) seen_weight[c.card_id] += strong_weight; // strong
            for (const auto &c : player.battle_zone) seen_weight[c.card_id] += strong_weight; // strong
            for (const auto &c : player.shield_zone) seen_weight[c.card_id] += strong_weight; // strong
            for (const auto &c : player.graveyard) seen_weight[c.card_id] += strong_weight; // strong
            for (const auto &c : player.deck) seen_weight[c.card_id] += deck_weight; // weak evidence
        }

        // Detect reveals: a card that was in prev_state.deck but now appears in hand/battle/shield/grave
        for (size_t pid = 0; pid < prev_state.players.size() && pid < state.players.size(); ++pid) {
            std::set<uint16_t> prev_deck_ids;
            for (const auto &c : prev_state.players[pid].deck) prev_deck_ids.insert(c.card_id);

            // collect visible zones in current state for this player
            std::set<uint16_t> curr_visible;
            for (const auto &c : state.players[pid].hand) curr_visible.insert(c.card_id);
            for (const auto &c : state.players[pid].battle_zone) curr_visible.insert(c.card_id);
            for (const auto &c : state.players[pid].shield_zone) curr_visible.insert(c.card_id);
            for (const auto &c : state.players[pid].graveyard) curr_visible.insert(c.card_id);

            // for intersection, add reveal weight
            for (auto id : prev_deck_ids) {
                if (curr_visible.find(id) != curr_visible.end()) {
                    seen_weight[id] += reveal_weight;
                }
            }
        }

        // Penalize probabilities for seen cards proportionally to weight, then renormalize
        for (auto &p : probs) {
            auto it = seen_weight.find(p.first);
            if (it != seen_weight.end() && it->second > 0.0f) {
                float weight = it->second;
                float multiplier = 1.0f / (1.0f + weight);
                p.second *= multiplier;
            }
        }

        // Renormalize
        float sum = 0.0f;
        for (const auto &p : probs) sum += p.second;
        if (sum <= 0.0f) return;
        for (auto &p : probs) p.second /= sum;
    }

    void set_reveal_weight(float w) { reveal_weight = w; }
    float get_reveal_weight() const { return reveal_weight; }

    // Return belief vector as probabilities in ascending card id order
    std::vector<float> get_vector() const {
        std::vector<float> out;
        out.reserve(probs.size());
        for (const auto &p : probs) out.push_back(p.second);
        return out;
    }

private:
    std::map<uint16_t, float> probs;
    float strong_weight = 1.0f;
    float deck_weight = 0.25f;
    float reveal_weight = 1.0f; // extra weight applied when a deck->visible transition is detected
};

} // namespace ai
} // namespace dm
