#pragma once

#include <vector>
#include <string>
#include <map>
#include <random>
#include <optional>
#include "core/game_state.hpp"
#include "core/card_def.hpp"

namespace dm::ai::inference {

    struct MetaDeck {
        std::string name;
        std::vector<dm::core::CardID> cards; // Should be sorted or multiset-like for easy comparison
    };

    class DeckInference {
    public:
        DeckInference();

        // Load decks from a JSON file
        void load_decks(const std::string& filepath);

        // Calculate probabilities for each deck based on observed game state
        // Returns a map of Deck Name -> Probability
        std::map<std::string, float> infer_probabilities(
            const dm::core::GameState& state,
            dm::core::PlayerID observer_id
        );

        // Get a pool of candidates for the hidden zones (Hand + Deck + Shield)
        // based on the inferred probabilities.
        // This samples ONE deck archetype based on the probability distribution,
        // then subtracts visible cards, and returns the remainder.
        std::vector<dm::core::CardID> sample_hidden_cards(
            const dm::core::GameState& state,
            dm::core::PlayerID observer_id,
            uint32_t seed
        );

    private:
        std::vector<MetaDeck> decks_;

        // Helper to count cards in a vector
        std::map<dm::core::CardID, int> count_cards(const std::vector<dm::core::CardID>& cards) const;

        // Helper to check if observed cards are compatible with a deck archetype
        bool is_compatible(const std::map<dm::core::CardID, int>& observed,
                           const std::map<dm::core::CardID, int>& archetype) const;
    };

}
