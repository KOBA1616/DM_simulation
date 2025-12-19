#include "token_converter.hpp"
#include <algorithm>
#include <iostream>

namespace dm::ai::encoders {

    void TokenConverter::append_card(std::vector<int>& tokens, const dm::core::CardInstance& card, bool visible) {
        if (!visible) {
            tokens.push_back(TOKEN_UNK);
            if (card.is_tapped) tokens.push_back(STATE_TAPPED);
            return;
        }

        int id_token = BASE_CARD_ID + card.card_id;
        tokens.push_back(id_token);

        if (card.is_tapped) tokens.push_back(STATE_TAPPED);
        if (card.summoning_sickness) tokens.push_back(STATE_SICK);
        if (card.is_face_down) tokens.push_back(STATE_FACE_DOWN);
    }

    void TokenConverter::append_zone(std::vector<int>& tokens, const std::vector<dm::core::CardInstance>& zone, int zone_token, bool visible) {
        tokens.push_back(zone_token);
        for (const auto& card : zone) {
            append_card(tokens, card, visible);
        }
        tokens.push_back(TOKEN_SEP);
    }

    void TokenConverter::append_command_history(std::vector<int>& tokens, const dm::core::GameState& state, int limit) {
        int start_idx = 0;
        int hist_size = state.command_history.size();
        if (limit > 0 && hist_size > limit) {
            start_idx = hist_size - limit;
        }
        if (start_idx < 0) start_idx = 0;

        for (int i = start_idx; i < hist_size; ++i) {
            auto& cmd = state.command_history[i];
            using namespace dm::engine::game_command;
            CommandType type = cmd->get_type();

            int cmd_token = BASE_COMMAND_MARKER;
            switch(type) {
                case CommandType::TRANSITION: cmd_token = CMD_TRANSITION; break;
                case CommandType::MUTATE: cmd_token = CMD_MUTATE; break;
                case CommandType::ATTACH: cmd_token = CMD_ATTACH; break;
                case CommandType::FLOW: cmd_token = CMD_FLOW; break;
                case CommandType::QUERY: cmd_token = CMD_QUERY; break;
                case CommandType::DECIDE: cmd_token = CMD_DECIDE; break;
                case CommandType::DECLARE_REACTION: cmd_token = CMD_REACTION; break;
                case CommandType::STAT: cmd_token = CMD_STAT; break;
                case CommandType::GAME_RESULT: cmd_token = CMD_RESULT; break;
            }
            tokens.push_back(cmd_token);

            if (type == CommandType::TRANSITION) {
                auto trans = std::dynamic_pointer_cast<TransitionCommand>(cmd);
                if (trans) {
                    const auto* inst = state.get_card_instance(trans->card_instance_id);
                    if (inst) {
                         tokens.push_back(BASE_CARD_ID + inst->card_id);
                    }
                }
            }
        }
    }

    std::vector<int> TokenConverter::encode_state(const dm::core::GameState& state, int max_len) {
        std::vector<int> tokens;
        tokens.reserve(1024);

        tokens.push_back(TOKEN_CLS);

        tokens.push_back(BASE_CONTEXT_MARKER + state.turn_number);
        tokens.push_back(BASE_CONTEXT_MARKER + 50 + (int)state.current_phase);
        tokens.push_back(TOKEN_SEP);

        const auto& p1 = state.players[0];
        append_zone(tokens, p1.hand, MARKER_HAND_P1, true);
        append_zone(tokens, p1.mana_zone, MARKER_MANA_P1, true);
        append_zone(tokens, p1.battle_zone, MARKER_BATTLE_P1, true);
        append_zone(tokens, p1.shield_zone, MARKER_SHIELD_P1, true);
        append_zone(tokens, p1.graveyard, MARKER_GRAVE_P1, true);

        if (state.players.size() > 1) {
            const auto& p2 = state.players[1];
            append_zone(tokens, p2.hand, MARKER_HAND_P2, false);
            append_zone(tokens, p2.mana_zone, MARKER_MANA_P2, true);
            append_zone(tokens, p2.battle_zone, MARKER_BATTLE_P2, true);
            append_zone(tokens, p2.shield_zone, MARKER_SHIELD_P2, false);
            append_zone(tokens, p2.graveyard, MARKER_GRAVE_P2, true);
        }

        append_command_history(tokens, state, 30);

        if (max_len > 0 && tokens.size() > max_len) {
            tokens.resize(max_len);
        }

        return tokens;
    }

}
