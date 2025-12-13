#include "tensor_converter.hpp"
#include "core/constants.hpp"
#include "core/card_def.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <map>
#include <iostream>

namespace dm::ai {

    using namespace dm::core;

    // --- Legacy V1 (ResNet) Implementation ---
    std::vector<float> TensorConverter::convert_to_tensor(
        const GameState& game_state,
        int player_view,
        const std::map<CardID, CardDefinition>& card_db,
        bool mask_opponent_hand
    ) {
        std::vector<float> tensor;
        tensor.reserve(INPUT_SIZE);

        // 1. Global Features
        tensor.push_back(static_cast<float>(game_state.turn_number) / static_cast<float>(TURN_LIMIT));
        tensor.push_back(static_cast<float>(game_state.current_phase) / 5.0f); // 6 phases 0-5
        tensor.push_back(game_state.active_player_id == player_view ? 1.0f : 0.0f);
        for(int i=0; i<7; ++i) tensor.push_back(0.0f); // Padding

        const Player& self = game_state.players[player_view];
        const Player& opp = game_state.players[1 - player_view];

        auto encode_card_feature = [&](CardID cid) -> float {
            if (!card_db.count(cid)) return 0.0f;
            const auto& def = card_db.at(cid);
            float pwr = 0.0f;
            if (def.power >= POWER_INFINITY) pwr = 1.0f;
            else pwr = std::min(def.power, 100) / 100.0f;
            float cost = std::min(def.cost, 20) / 20.0f;

            uint8_t civv = 0;
            for (auto c : def.civilizations) civv |= static_cast<uint8_t>(c);
            int pop = 0;
            for (int b = 0; b < 8; ++b) if (civv & (1u << b)) ++pop;
            float civ_norm = static_cast<float>(pop) / 6.0f;
            float type_flag = (def.type == CardType::CREATURE) ? 1.0f : 0.0f;
            return pwr * 0.7f + cost * 0.15f + civ_norm * 0.1f + type_flag * 0.05f;
        };

        auto encode_player = [&](const Player& p, bool is_self_or_full_info) {
            tensor.push_back(static_cast<float>(p.hand.size()) / static_cast<float>(MAX_HAND_SIZE));
            for (int i = 0; i < MAX_HAND_SIZE; ++i) {
                if (i < (int)p.hand.size() && is_self_or_full_info) {
                    tensor.push_back(encode_card_feature(p.hand[i].card_id));
                } else {
                    tensor.push_back(0.0f);
                }
            }
            int civ_counts[7] = {0};
            for (const auto& c : p.mana_zone) {
                if (card_db.count(c.card_id)) {
                    uint8_t val = 0;
                    for (auto civ : card_db.at(c.card_id).civilizations) val |= static_cast<uint8_t>(civ);
                    if (val & static_cast<uint8_t>(Civilization::LIGHT)) civ_counts[1]++;
                    if (val & static_cast<uint8_t>(Civilization::WATER)) civ_counts[2]++;
                    if (val & static_cast<uint8_t>(Civilization::DARKNESS)) civ_counts[3]++;
                    if (val & static_cast<uint8_t>(Civilization::FIRE)) civ_counts[4]++;
                    if (val & static_cast<uint8_t>(Civilization::NATURE)) civ_counts[5]++;
                    if (val & static_cast<uint8_t>(Civilization::ZERO)) civ_counts[6]++;
                }
            }
            for(int i=1; i<=6; ++i) tensor.push_back(static_cast<float>(civ_counts[i]) / static_cast<float>(MAX_MANA_SIZE));
            for (int i = 0; i < MAX_BATTLE_SIZE; ++i) {
                if (i < (int)p.battle_zone.size()) {
                    const auto& c = p.battle_zone[i];
                    tensor.push_back(encode_card_feature(c.card_id));
                    tensor.push_back(c.is_tapped ? 1.0f : 0.0f);
                    tensor.push_back(c.summoning_sickness ? 1.0f : 0.0f);
                } else {
                    tensor.push_back(0.0f);
                    tensor.push_back(0.0f);
                    tensor.push_back(0.0f);
                }
            }
            tensor.push_back(static_cast<float>(p.shield_zone.size()));
            for (int i = 0; i < MAX_GRAVE_SEARCH; ++i) {
                if (i < (int)p.graveyard.size()) {
                    int idx = (int)p.graveyard.size() - 1 - i;
                    if (idx >= 0) tensor.push_back(encode_card_feature(p.graveyard[idx].card_id));
                    else tensor.push_back(0.0f);
                } else {
                    tensor.push_back(0.0f);
                }
            }
        };

        encode_player(self, true);
        encode_player(opp, !mask_opponent_hand);
        return tensor;
    }

    std::vector<float> TensorConverter::convert_batch_flat(
        const std::vector<GameState>& states,
        const std::map<CardID, CardDefinition>& card_db,
        bool mask_opponent_hand
    ) {
        std::vector<float> batch_tensor;
        batch_tensor.reserve(states.size() * INPUT_SIZE);
        for (const auto& state : states) {
            std::vector<float> t = convert_to_tensor(state, state.active_player_id, card_db, mask_opponent_hand);
            batch_tensor.insert(batch_tensor.end(), t.begin(), t.end());
        }
        return batch_tensor;
    }

    // --- Phase 4 V2 (Transformer) Implementation ---

    std::vector<long> TensorConverter::convert_to_sequence(
        const GameState& game_state,
        int player_view,
        const std::map<CardID, CardDefinition>& card_db,
        bool mask_opponent_hand
    ) {
        std::vector<long> seq;
        seq.reserve(MAX_SEQ_LEN);

        // Helper to add token safely
        auto add_token = [&](long token) {
            if (seq.size() < MAX_SEQ_LEN) {
                seq.push_back(token);
            }
        };

        auto add_card = [&](CardID cid) {
            // Map CardID to Token. Assume CardID + Offset.
            // Check vocab size limits in real app.
            add_token(TOKEN_CARD_OFFSET + static_cast<long>(cid));
        };

        // 1. Global
        add_token(TOKEN_GLOBAL_START);
        // Encode Turn and Phase as tokens?
        // For simplicity, we might just rely on the separator structure or add specific tokens.
        // Here we just put tokens for structure.

        const Player& self = game_state.players[player_view];
        const Player& opp = game_state.players[1 - player_view];

        // 2. Self Hand
        add_token(TOKEN_SELF_HAND_START);
        for (const auto& c : self.hand) add_card(c.card_id);

        // 3. Self Mana
        add_token(TOKEN_SELF_MANA_START);
        for (const auto& c : self.mana_zone) add_card(c.card_id);

        // 4. Self Battle
        add_token(TOKEN_SELF_BATTLE_START);
        for (const auto& c : self.battle_zone) {
            add_card(c.card_id);
            // Status (tapped, sick) could be separate tokens or added to ID
        }

        // 5. Self Shields
        add_token(TOKEN_SELF_SHIELD_START);
        for (const auto& c : self.shield_zone) add_card(c.card_id);

        // 6. Opponent
        add_token(TOKEN_OPP_HAND_START);
        if (!mask_opponent_hand) {
            for (const auto& c : opp.hand) add_card(c.card_id);
        } else {
             // Add PAD or specialized UNKNOWN token for each card count?
             for (size_t i=0; i<opp.hand.size(); ++i) add_token(TOKEN_PAD);
        }

        add_token(TOKEN_OPP_MANA_START);
        for (const auto& c : opp.mana_zone) add_card(c.card_id);

        add_token(TOKEN_OPP_BATTLE_START);
        for (const auto& c : opp.battle_zone) add_card(c.card_id);

        // Pad remaining
        while(seq.size() < MAX_SEQ_LEN) {
            seq.push_back(TOKEN_PAD);
        }

        return seq;
    }

    std::vector<long> TensorConverter::convert_batch_sequence(
        const std::vector<GameState>& states,
        const std::map<CardID, CardDefinition>& card_db,
        bool mask_opponent_hand
    ) {
        std::vector<long> batch_seq;
        batch_seq.reserve(states.size() * MAX_SEQ_LEN);
        for (const auto& state : states) {
            std::vector<long> s = convert_to_sequence(state, state.active_player_id, card_db, mask_opponent_hand);
            batch_seq.insert(batch_seq.end(), s.begin(), s.end());
        }
        return batch_seq;
    }

}
