#pragma once

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "deck_inference.hpp"
#include <vector>
#include <map>
#include <memory>
#include <random>

namespace dm::ai::inference {

    // Perfect Information Monte Carlo Generator
    // Generates determinized GameStates from an imperfect information state
    // by sampling hidden information.
    class PimcGenerator {
    public:
        // Constructor taking shared_ptr to ensure lifetime safety
        PimcGenerator(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db);

        // Set the deck inference model
        void set_inference_model(std::shared_ptr<DeckInference> inference);

        // Generate a fully determinized GameState (for the observer)
        // This fills the opponent's hand, deck, and shields with sampled cards.
        dm::core::GameState generate_determinized_state(
            const dm::core::GameState& observed_state,
            dm::core::PlayerID observer_id,
            uint32_t seed
        );

        // Static helper for direct usage with a known candidate pool (SAFE: does not store reference)
        static dm::core::GameState generate_determinized_state(
            const dm::core::GameState& observed_state,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            dm::core::PlayerID observer_id,
            const std::vector<dm::core::CardID>& opponent_deck_candidates,
            uint32_t seed
        );

    private:
        std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db_;
        std::shared_ptr<DeckInference> inference_;
    };

}
