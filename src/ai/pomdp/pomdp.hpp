#pragma once

#include <vector>
#include <map>
#include <string>
#include <memory>
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "ai/inference/deck_inference.hpp"
#include "ai/inference/pimc_generator.hpp"

namespace dm::ai {

class POMDPInference {
public:
    POMDPInference();

    // Initialize with card database and path to meta decks
    void initialize(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::string& meta_deck_path);

    // Update internal belief given an observed GameState
    // For this simplified version, this is stateless/stateless-ish,
    // relying on DeckInference to re-evaluate probabilities from the current state each time.
    void update_belief(const dm::core::GameState& state);

    // Sample a concrete GameState from the belief distribution
    // This performs Determinization (filling hidden information)
    dm::core::GameState sample_state(const dm::core::GameState& observation, uint32_t seed);

    // Get inferred probabilities for each meta deck
    std::map<std::string, float> get_deck_probabilities(const dm::core::GameState& state, dm::core::PlayerID observer_id);

private:
    std::unique_ptr<dm::ai::inference::DeckInference> deck_inference_;
    std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
    bool initialized_ = false;
};

} // namespace dm::ai
