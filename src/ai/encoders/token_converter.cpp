#include "token_converter.hpp"
#include "engine/systems/card/card_registry.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace dm::ai::encoders {

    int TokenConverter::get_vocab_size() {
        return VOCAB_SIZE;
    }

    int TokenConverter::bucket_val(int val, int max_val) {
        if (val < 0) return 0;
        if (val >= max_val) return max_val;
        return val;
    }

    std::vector<int> TokenConverter::encode_state(const core::GameState& state, int perspective, int max_len) {
        const auto& db = dm::engine::CardRegistry::get_all_definitions();

        GameStateTokens tokens = tokenize_state(state, db);

        std::vector<int> flat;
        flat.reserve(2048);

        // 1. Global Section
        flat.push_back(TOKEN_CLS);
        flat.insert(flat.end(), tokens.global_features.begin(), tokens.global_features.end());

        // 2. Board Section
        flat.push_back(TOKEN_SEP_BOARD);
        for(auto& vec : tokens.board_tokens) {
             flat.insert(flat.end(), vec.begin(), vec.end());
        }

        // 3. History Section
        flat.push_back(TOKEN_SEP_HISTORY);
        for(auto& vec : tokens.history_tokens) {
            flat.insert(flat.end(), vec.begin(), vec.end());
        }

        // Truncate
        if (max_len > 0 && flat.size() > max_len) {
            flat.resize(max_len);
        }

        return flat;
    }

    GameStateTokens TokenConverter::tokenize_state(const core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        GameStateTokens tokens;

        // 1. Global Features (Offset by OFFSET_GLOBAL)
        // Ensure values don't overflow into other ranges
        tokens.global_features.push_back(OFFSET_GLOBAL + bucket_val(state.turn_number, 50));
        tokens.global_features.push_back(OFFSET_GLOBAL + 51 + bucket_val(static_cast<int>(state.active_player_id), 2));
        tokens.global_features.push_back(OFFSET_GLOBAL + 54 + bucket_val(static_cast<int>(state.current_phase), 10));

        auto add_player_stats = [&](const core::Player& p) {
            tokens.global_features.push_back(OFFSET_GLOBAL + 70 + bucket_val(p.mana_zone.size(), 20));
            tokens.global_features.push_back(OFFSET_GLOBAL + 90 + bucket_val(p.shield_zone.size(), 10));
            tokens.global_features.push_back(OFFSET_GLOBAL + 100 + bucket_val(p.hand.size(), 20));
        };

        if (state.players.size() > 0) add_player_stats(state.players[0]);
        else tokens.global_features.insert(tokens.global_features.end(), {OFFSET_GLOBAL, OFFSET_GLOBAL, OFFSET_GLOBAL});

        if (state.players.size() > 1) add_player_stats(state.players[1]);
        else tokens.global_features.insert(tokens.global_features.end(), {OFFSET_GLOBAL, OFFSET_GLOBAL, OFFSET_GLOBAL});

        // 2. All Zones
        for (const auto& player : state.players) {
            auto process_zone = [&](const std::vector<core::CardInstance>& zone) {
                for (const auto& card : zone) {
                    if (card_db.count(card.card_id)) {
                        tokens.board_tokens.push_back(tokenize_card_instance(card, card_db.at(card.card_id)));
                    } else {
                        core::CardDefinition dummy; dummy.id = card.card_id;
                        tokens.board_tokens.push_back(tokenize_card_instance(card, dummy));
                    }
                }
            };
            process_zone(player.battle_zone);
            process_zone(player.hand);
            process_zone(player.mana_zone);
            process_zone(player.shield_zone);
            process_zone(player.graveyard);
        }

        if (tokens.board_tokens.size() > MAX_BOARD_ENTITIES) {
            tokens.board_tokens.resize(MAX_BOARD_ENTITIES);
        }

        // 3. History
        int start_idx = std::max(0, (int)state.command_history.size() - MAX_HISTORY_LEN);
        for (int i = start_idx; i < state.command_history.size(); ++i) {
            tokens.history_tokens.push_back(tokenize_command(state.command_history[i]));
        }

        return tokens;
    }

    std::vector<int> TokenConverter::tokenize_card_instance(const core::CardInstance& card, const core::CardDefinition& def) {
        std::vector<int> feat;
        feat.reserve(8);

        // [0] Card ID (Offset to avoid collision)
        // Clamp ID to range to avoid overflow
        int cid = card.card_id;
        if (cid > 2500) cid = 2500; // Unknown/OOB
        feat.push_back(OFFSET_CARD_ID + cid);

        // [1] Tapped
        feat.push_back(card.is_tapped ? TOKEN_TAPPED : TOKEN_UNTAPPED);

        // [2] Sickness
        feat.push_back(card.summoning_sickness ? TOKEN_SICK : TOKEN_NOT_SICK);

        // [3] Power (Bucketed + Offset)
        // Reuse OFFSET_STATUS + 10 for power buckets? Or just use OFFSET_GLOBAL if distinct?
        // Let's use specific status range: OFFSET_STATUS + 20 + bucket
        int p_bucket = bucket_val(def.power / 1000, 30);
        feat.push_back(OFFSET_STATUS + 20 + p_bucket);

        // [4] Cost
        int c_bucket = bucket_val(def.cost, 20);
        feat.push_back(OFFSET_STATUS + 60 + c_bucket);

        // [5] Civilization (Bitmask mapped to token?)
        // Simply cast to int and offset, max 32.
        int civ_mask = 0;
        for (auto c : def.civilizations) civ_mask |= static_cast<int>(c);
        feat.push_back(OFFSET_STATUS + 90 + (civ_mask & 0x3F));

        return feat;
    }

    std::vector<int> TokenConverter::tokenize_command(const std::shared_ptr<dm::engine::game_command::GameCommand>& cmd) {
        std::vector<int> feat;
        feat.push_back(OFFSET_ACTION + static_cast<int>(cmd->get_type()));

        using namespace dm::engine::game_command;

        // Add basic args as buckets in Global range or special ranges
        // Simplify for now: Just Type + maybe generic value bucket
        feat.push_back(OFFSET_GLOBAL); // Placeholder arg

        return feat;
    }

}
