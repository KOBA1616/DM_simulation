#include "deck_inference.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace dm::ai::inference {

    DeckInference::DeckInference() {}

    void DeckInference::load_decks(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[DeckInference] Error: Could not open file " << filepath << std::endl;
            return;
        }

        json j;
        try {
            file >> j;
        } catch (const std::exception& e) {
             std::cerr << "[DeckInference] JSON parse error: " << e.what() << std::endl;
             return;
        }

        decks_.clear();
        if (j.contains("decks")) {
            for (const auto& deck_entry : j["decks"]) {
                MetaDeck deck;
                deck.name = deck_entry.value("name", "Unknown");
                if (deck_entry.contains("cards")) {
                    for (int card_id : deck_entry["cards"]) {
                        deck.cards.push_back(static_cast<dm::core::CardID>(card_id));
                    }
                }
                decks_.push_back(deck);
            }
        }
        std::cout << "[DeckInference] Loaded " << decks_.size() << " decks from " << filepath << std::endl;
    }

    std::map<dm::core::CardID, int> DeckInference::count_cards(const std::vector<dm::core::CardID>& cards) const {
        std::map<dm::core::CardID, int> counts;
        for (const auto& id : cards) {
            counts[id]++;
        }
        return counts;
    }

    bool DeckInference::is_compatible(const std::map<dm::core::CardID, int>& observed,
                                      const std::map<dm::core::CardID, int>& archetype) const {
        for (const auto& [id, count] : observed) {
            auto it = archetype.find(id);
            if (it == archetype.end()) {
                // Observed card not in archetype -> Incompatible
                // Unless ID is 0 or some dummy?
                if (id == 0) continue;
                return false;
            }
            if (count > it->second) {
                // Observed more copies than in archetype -> Incompatible
                return false;
            }
        }
        return true;
    }

    std::map<std::string, float> DeckInference::infer_probabilities(
        const dm::core::GameState& state,
        dm::core::PlayerID observer_id
    ) {
        dm::core::PlayerID opponent_id = 1 - observer_id;
        const auto& opponent = state.players[opponent_id];

        // 1. Collect observed cards from opponent
        std::vector<dm::core::CardID> observed_cards;

        // Public Zones: Mana, Battle, Graveyard
        for (const auto& card : opponent.mana_zone) observed_cards.push_back(card.card_id);
        for (const auto& card : opponent.battle_zone) observed_cards.push_back(card.card_id);
        for (const auto& card : opponent.graveyard) observed_cards.push_back(card.card_id);

        // Hand: Check for revealed cards.
        // In GameState, if we are the observer, hidden cards in opponent hand might be masked (ID 0)
        // or effectively hidden. If the engine provided true state, we must check if we *should* know it.
        // Assuming the input state is already "observed" (i.e., masked), we just use valid IDs.
        // If the ID is valid (>0), it means it's revealed (e.g. by a specific effect or just face up).
        for (const auto& card : opponent.hand) {
            if (card.card_id > 0) { // Assuming 0 is dummy/masked
                 observed_cards.push_back(card.card_id);
            }
        }

        // Shields: Similarly, face up shields might be visible.
         for (const auto& card : opponent.shield_zone) {
            if (card.card_id > 0 && !card.is_face_down) {
                 observed_cards.push_back(card.card_id);
            }
        }

        auto observed_counts = count_cards(observed_cards);

        // 2. Evaluate decks
        std::map<std::string, float> probabilities;
        std::vector<float> likelihoods;

        for (const auto& deck : decks_) {
            auto deck_counts = count_cards(deck.cards);
            if (is_compatible(observed_counts, deck_counts)) {
                likelihoods.push_back(1.0f); // Uniform prior, hard constraint
            } else {
                likelihoods.push_back(0.0f);
            }
        }

        // 3. Normalize
        float total_likelihood = std::accumulate(likelihoods.begin(), likelihoods.end(), 0.0f);

        if (total_likelihood > 0) {
            for (size_t i = 0; i < decks_.size(); ++i) {
                probabilities[decks_[i].name] = likelihoods[i] / total_likelihood;
            }
        } else {
            // No deck is compatible? This shouldn't happen unless the meta is incomplete.
            // Fallback: Uniform distribution over all decks? Or just return empty/error?
            // Let's fallback to uniform distribution for robustness.
             for (const auto& deck : decks_) {
                probabilities[deck.name] = 1.0f / decks_.size();
            }
            // Or maybe the opponent is playing an unknown deck.
        }

        return probabilities;
    }

    std::vector<dm::core::CardID> DeckInference::sample_hidden_cards(
        const dm::core::GameState& state,
        dm::core::PlayerID observer_id,
        uint32_t seed
    ) {
        std::map<std::string, float> probs = infer_probabilities(state, observer_id);

        // 1. Sample a deck archetype
        std::vector<float> weights;
        std::vector<const MetaDeck*> valid_decks;

        for (const auto& deck : decks_) {
            if (probs[deck.name] > 0) {
                weights.push_back(probs[deck.name]);
                valid_decks.push_back(&deck);
            }
        }

        if (valid_decks.empty()) {
            // Fallback: Return empty or random?
            // If we have no decks, we can't infer anything.
            return {};
        }

        std::mt19937 rng(seed);
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        int chosen_idx = dist(rng);
        const MetaDeck* chosen_deck = valid_decks[chosen_idx];

        // 2. Subtract observed cards
        auto deck_counts = count_cards(chosen_deck->cards);

        // Count observed cards again
        dm::core::PlayerID opponent_id = 1 - observer_id;
        const auto& opponent = state.players[opponent_id];

        std::vector<dm::core::CardID> observed_cards;
        for (const auto& card : opponent.mana_zone) observed_cards.push_back(card.card_id);
        for (const auto& card : opponent.battle_zone) observed_cards.push_back(card.card_id);
        for (const auto& card : opponent.graveyard) observed_cards.push_back(card.card_id);
        for (const auto& card : opponent.hand) {
            if (card.card_id > 0) observed_cards.push_back(card.card_id);
        }
        for (const auto& card : opponent.shield_zone) {
             if (card.card_id > 0 && !card.is_face_down) observed_cards.push_back(card.card_id);
        }

        for (auto id : observed_cards) {
            if (deck_counts.count(id) && deck_counts[id] > 0) {
                deck_counts[id]--;
            }
            // If observed card not in deck, we ignore it (it's consistent with "incompatible" logic already handled,
            // or we are in the fallback case where we just do best effort).
        }

        // 3. Construct remaining pool
        std::vector<dm::core::CardID> pool;
        for (const auto& [id, count] : deck_counts) {
            for (int i = 0; i < count; ++i) {
                pool.push_back(id);
            }
        }

        return pool;
    }

}
