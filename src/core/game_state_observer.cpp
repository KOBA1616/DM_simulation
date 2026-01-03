#include "core/game_state.hpp"

namespace dm::core {

    GameState GameState::create_observer_view(PlayerID observer_id) const {
        GameState view = this->clone();

        PlayerID opponent_id = 1 - observer_id;

        // Ensure opponent_id is valid
        if (opponent_id >= 0 && opponent_id < view.players.size()) {
            auto& opp = view.players[opponent_id];

            // Mask Hand
            // Rules: Observers cannot see opponent's hand.
            // Exception: Cards explicitly revealed (e.g. by effect) might be visible,
            // but the engine currently relies on is_face_down or similar flags.
            // For PIMC, we mask ALL cards in hand unless we have a specific 'revealed' flag.
            // Currently, assuming all hand cards are hidden.
            for (auto& card : opp.hand) {
                // If the engine supported 'revealed' flags, we would check them here.
                // Since it doesn't seem to explicitly track 'revealed' state for hand cards
                // (other than is_face_down which is usually for shields/battle),
                // we mask everything.
                // NOTE: If an effect reveals a card, it might temporarily move it to a buffer or
                // keep it in hand. If it's in hand, we assume it's hidden again after the effect resolves
                // unless tracked.
                card.card_id = 0;
                // We keep other properties like 'tapped' (irrelevant for hand) or 'id' (instance id).
                // Instance ID is preserved to track object continuity.
            }

            // Mask Deck
            // Deck is always hidden.
            for (auto& card : opp.deck) {
                card.card_id = 0;
            }

            // Mask Shields
            // Shields are hidden unless face up.
            for (auto& card : opp.shield_zone) {
                if (card.is_face_down) {
                    card.card_id = 0;
                }
                // If face up (shield trigger check or explicit effect), we keep the ID.
            }

            // Mask Own Deck?
            // Players technically know their deck content by deduction (Initial - Visible),
            // but the order is unknown.
            // For PIMC, we usually treat the Deck Order as hidden.
            // Masking the IDs in the deck prevents the AI from "peeking" the topdeck.
            auto& self = view.players[observer_id];
            for (auto& card : self.deck) {
                 card.card_id = 0;
            }
        }

        return view;
    }

}
