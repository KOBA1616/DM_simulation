#include "pimc_generator.hpp"
#include <algorithm>
#include <iostream>

namespace dm::ai::inference {

    dm::core::GameState PIMCGenerator::generate_determinized_state(
        const dm::core::GameState& observation,
        const std::map<dm::core::CardID, dm::core::CardDefinition>& /*card_db*/,
        dm::core::PlayerID observer_id,
        const std::vector<dm::core::CardID>& opponent_deck_candidates,
        uint32_t seed
    ) {
        dm::core::GameState state = observation;

        // We need to randomize the hidden information for the OTHER player.
        dm::core::PlayerID opponent_id = 1 - observer_id;
        dm::core::Player& opponent = state.players[opponent_id];

        // 1. Identify hidden zones and counts
        // Hidden zones: Hand (unless revealed), Shield Zone, Deck.
        // Note: Some cards in hand might be revealed (e.g. "Simulacrum" logic),
        // but for standard PIMC we usually assume "Hand" is a hidden zone block.
        // However, if we want to respect "known" cards in hand, we need a mechanism.
        // For this implementation, we assume ALL cards in opponent hand, shield, deck are to be sampled
        // from the candidate pool, preserving only their count and instance_ids.

        // Important: If we are replacing cards, we must ensure we don't invalidate instance_ids
        // that might be referenced by pending effects.
        // PIMC is typically run at a decision point (start of turn, or when priority is passed),
        // where usually there are no pending effects targeting specific hidden cards.

        int hand_count = opponent.hand.size();
        int shield_count = opponent.shield_zone.size();
        int deck_count = opponent.deck.size();

        size_t total_needed = hand_count + shield_count + deck_count;

        std::vector<dm::core::CardID> pool = opponent_deck_candidates;

        // Validation: If pool is smaller than needed, we must handle it.
        // Fill with dummy cards (ID 0) if insufficient.
        if (pool.size() < total_needed) {
            // Log warning?
            // std::cerr << "PIMC Warning: Insufficient candidates. Needed " << total_needed
            //           << ", have " << pool.size() << ". Filling with ID 0." << std::endl;
            while (pool.size() < total_needed) {
                pool.push_back(0); // Assuming 0 is valid or dummy
            }
        }

        // 2. Shuffle the pool
        std::mt19937 rng(seed);
        std::shuffle(pool.begin(), pool.end(), rng);

        // 3. Distribute to zones
        int pool_idx = 0;

        // Fill Hand
        for (auto& card : opponent.hand) {
            card.card_id = pool[pool_idx++];
            // Keep instance_id, tapped state, etc.
            // Reset known flags if necessary? e.g. if we knew it was a specific card, we are overwriting it.
        }

        // Fill Shield Zone
        for (auto& card : opponent.shield_zone) {
            card.card_id = pool[pool_idx++];
            // Shields are usually face down.
        }

        // Fill Deck
        for (auto& card : opponent.deck) {
            card.card_id = pool[pool_idx++];
        }

        // Note: We do NOT touch observer's zones or public zones (Mana, Grave, Battle).

        return state;
    }

}
