#include "tensor_converter.hpp"
#include "../../core/constants.hpp"
#include "../../core/card_def.hpp"
#include "../../core/types.hpp"
#include <algorithm>
#include <map>
#include <iostream>

namespace dm::ai {

    using namespace dm::core;

    std::vector<float> TensorConverter::convert_to_tensor(const GameState& game_state, int player_view, const std::map<CardID, CardDefinition>& card_db) {
        std::vector<float> tensor;
        tensor.reserve(INPUT_SIZE);

        // Debug
        // std::cout << "DB Size: " << card_db.size() << std::endl;

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
                tensor.push_back(static_cast<float>(p.hand.size()));
            }

            // Mana (Civ counts)
            int civ_counts[7] = {0}; // 0=unused, 1=L, 2=W, 3=D, 4=F, 5=N, 6=Z
            for (const auto& c : p.mana_zone) {
                if (card_db.count(c.card_id)) {
                    Civilization civ = card_db.at(c.card_id).civilization;
                    uint8_t val = static_cast<uint8_t>(civ);
                    
                    if (val & static_cast<uint8_t>(Civilization::LIGHT)) civ_counts[1]++;
                    if (val & static_cast<uint8_t>(Civilization::WATER)) civ_counts[2]++;
                    if (val & static_cast<uint8_t>(Civilization::DARKNESS)) civ_counts[3]++;
                    if (val & static_cast<uint8_t>(Civilization::FIRE)) civ_counts[4]++;
                    if (val & static_cast<uint8_t>(Civilization::NATURE)) civ_counts[5]++;
                    if (val & static_cast<uint8_t>(Civilization::ZERO)) civ_counts[6]++;
                }
            }
            // Push counts for L, W, D, F, N, Z (Indices 1-6)
            for(int i=1; i<=6; ++i) {
                tensor.push_back(static_cast<float>(civ_counts[i]));
            }

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
