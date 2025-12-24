#include "ai/pomdp/pomdp.hpp"
#include <iostream>

namespace dm::ai {

POMDPInference::POMDPInference() {
    deck_inference_ = std::make_unique<dm::ai::inference::DeckInference>();
}

void POMDPInference::initialize(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::string& meta_deck_path) {
    card_db_ = card_db;
    deck_inference_->load_decks(meta_deck_path);
    initialized_ = true;
}

void POMDPInference::update_belief(const dm::core::GameState& /*state*/) {
    // In this implementation, belief is recalculated on demand in sample_state/get_deck_probabilities.
    // However, if we wanted to maintain a stateful particle filter, we would update it here.
    if (!initialized_) {
        std::cerr << "[POMDPInference] Warning: Not initialized." << std::endl;
    }
}

dm::core::GameState POMDPInference::sample_state(const dm::core::GameState& observation, uint32_t seed) {
    if (!initialized_) {
        std::cerr << "[POMDPInference] Error: Not initialized. Returning observation as is." << std::endl;
        return observation.clone();
    }

    // 1. Identify observer
    // The observation is from the perspective of active_player_id usually, or specifically requested.
    // However, GameState doesn't explicitly store "who is observing".
    // We assume the non-active player's hand is hidden if we are the active player.
    // For MCTS, we usually want to determinize the *opponent's* hidden info from *our* perspective.
    // Let's assume the observer is the player who is *not* having their cards randomized.
    // But PIMCGenerator::generate_determinized_state takes observer_id.

    // If it's my turn (active_player), I am observing. I want to guess opponent's hand.
    dm::core::PlayerID observer_id = observation.active_player_id;

    // 2. Sample hidden cards for the opponent
    std::vector<dm::core::CardID> opponent_candidates = deck_inference_->sample_hidden_cards(observation, observer_id, seed);

    // 3. Generate determinized state
    return dm::ai::inference::PIMCGenerator::generate_determinized_state(
        observation,
        card_db_,
        observer_id,
        opponent_candidates,
        seed
    );
}

std::map<std::string, float> POMDPInference::get_deck_probabilities(const dm::core::GameState& state, dm::core::PlayerID observer_id) {
    if (!initialized_) {
        return {};
    }
    return deck_inference_->infer_probabilities(state, observer_id);
}

} // namespace dm::ai
