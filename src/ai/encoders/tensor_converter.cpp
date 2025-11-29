#include "tensor_converter.hpp"
#include "../../core/constants.hpp"
#include <algorithm>

namespace dm::ai {

    using namespace dm::core;

    std::vector<float> TensorConverter::convert_to_tensor(const GameState& game_state, int player_view) {
        std::vector<float> tensor;
        tensor.reserve(INPUT_SIZE);

        // 1. Global Features
        tensor.push_back(static_cast<float>(game_state.turn_number) / static_cast<float>(TURN_LIMIT));
        tensor.push_back(static_cast<float>(game_state.current_phase) / 5.0f); // 6 phases 0-5
        tensor.push_back(game_state.active_player_id == player_view ? 1.0f : 0.0f);
        // Padding for global
        for(int i=0; i<7; ++i) tensor.push_back(0.0f);

        const Player& self = game_state.players[player_view];
        const Player& opp = game_state.players[1 - player_view];

        auto encode_player = [&](const Player& p, bool is_self) {
            // Hand
            if (is_self) {
                for (int i = 0; i < MAX_HAND_SIZE; ++i) {
                    if (i < (int)p.hand.size()) {
                        tensor.push_back(static_cast<float>(p.hand[i].card_id));
                    } else {
                        tensor.push_back(0.0f);
                    }
                }
            } else {
                // Opponent hand is masked, just count?
                // But we allocated space for slots in INPUT_SIZE calculation above?
                // "Opp: Hand(1)" in comment.
                // Wait, my calculation: (20 + ... ) for Self, (1 + ...) for Opp.
                // So for Opp, we just push count.
                tensor.push_back(static_cast<float>(p.hand.size()));
            }

            // Mana (Civ counts)
            int civ_counts[6] = {0}; // L, W, D, F, N, Z
            for (const auto& c : p.mana_zone) {
                // We need CardDefinition to know civ. 
                // But GameState doesn't have CardDB reference inside easily?
                // We might need to pass CardDB or store Civ in CardInstance?
                // For now, let's assume we can't get Civ without DB.
                // But TensorConverter usually needs DB.
                // Let's assume we just pass ID for now or 0 if we can't look it up.
                // Or better, CardInstance should probably cache basic info if performance is key.
                // For this implementation, I'll just count raw number of cards in mana for now, 
                // OR I need to change signature to accept DB.
                // Let's change signature in next step if needed. 
                // For now, just push 0s or raw IDs?
                // Spec says "Mana: 文明別枚数カウントに圧縮".
                // I will assume I can't do it without DB.
                // Let's just output raw IDs for Mana for now (up to 20?)
                // But I defined INPUT_SIZE based on counts.
                // Let's just push 0s for now and fix later or assume passed DB.
            }
            // Placeholder for Mana Civs (6 floats)
            for(int i=0; i<6; ++i) tensor.push_back(0.0f); 

            // Battle Zone
            for (int i = 0; i < MAX_BATTLE_SIZE; ++i) {
                if (i < (int)p.battle_zone.size()) {
                    const auto& c = p.battle_zone[i];
                    tensor.push_back(static_cast<float>(c.card_id));
                    tensor.push_back(c.is_tapped ? 1.0f : 0.0f);
                    tensor.push_back(c.summoning_sickness ? 1.0f : 0.0f);
                } else {
                    tensor.push_back(0.0f);
                    tensor.push_back(0.0f);
                    tensor.push_back(0.0f);
                }
            }

            // Shield Count
            tensor.push_back(static_cast<float>(p.shield_zone.size()));

            // Graveyard (Latest 20)
            for (int i = 0; i < MAX_GRAVE_SEARCH; ++i) {
                if (i < (int)p.graveyard.size()) {
                    // Latest is at back?
                    int idx = (int)p.graveyard.size() - 1 - i;
                    if (idx >= 0) {
                        tensor.push_back(static_cast<float>(p.graveyard[idx].card_id));
                    } else {
                        tensor.push_back(0.0f);
                    }
                } else {
                    tensor.push_back(0.0f);
                }
            }
        };

        encode_player(self, true);
        encode_player(opp, false);

        return tensor;
    }

}
