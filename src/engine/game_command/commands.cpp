#include "commands.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace dm::engine::game_command {

    // Helper to get mutable zone vector
    static std::vector<core::CardInstance>* get_zone_vector(core::GameState& state, int player_id, int zone_idx) {
        if (player_id < 0 || player_id > 1) return nullptr;
        core::Zone z = static_cast<core::Zone>(zone_idx);

        switch (z) {
            case core::Zone::HAND: return &state.players[player_id].hand;
            case core::Zone::MANA: return &state.players[player_id].mana_zone;
            case core::Zone::BATTLE: return &state.players[player_id].battle_zone;
            case core::Zone::GRAVEYARD: return &state.players[player_id].graveyard;
            case core::Zone::SHIELD: return &state.players[player_id].shield_zone;
            case core::Zone::DECK: return &state.players[player_id].deck;
            case core::Zone::HYPER_SPATIAL: return &state.players[player_id].hyper_spatial_zone;
            case core::Zone::GR_DECK: return &state.players[player_id].gr_deck;
            case core::Zone::STACK: return &state.stack_zone;
            case core::Zone::BUFFER: return &state.players[player_id].effect_buffer;
            default: return nullptr;
        }
    }

    // --- TransitionCommand ---

    TransitionCommand::TransitionCommand(int instance_id, int source_player, int source_zone,
                  int dest_player, int dest_zone, int dest_index)
        : instance_id_(instance_id), source_player_(source_player), source_zone_(source_zone),
          dest_player_(dest_player), dest_zone_(dest_zone), dest_index_(dest_index),
          previous_index_(-1), previous_owner_(-1) {}

    TransitionCommand::TransitionCommand(int instance_id, int source_zone, int dest_zone, int player_id, int dest_index)
        : instance_id_(instance_id), source_player_(player_id), source_zone_(source_zone),
          dest_player_(player_id), dest_zone_(dest_zone), dest_index_(dest_index),
          previous_index_(-1), previous_owner_(-1) {}

    void TransitionCommand::execute(core::GameState& state) {
        auto* src_vec = get_zone_vector(state, source_player_, source_zone_);
        auto* dest_vec = get_zone_vector(state, dest_player_, dest_zone_);

        if (!src_vec || !dest_vec) return;

        auto it = std::find_if(src_vec->begin(), src_vec->end(),
            [this](const core::CardInstance& c) { return c.instance_id == instance_id_; });

        if (it == src_vec->end()) return;

        previous_index_ = std::distance(src_vec->begin(), it);
        core::CardInstance card = *it;

        was_tapped_ = card.is_tapped;
        was_face_down_ = card.is_face_down;
        previous_owner_ = card.owner;

        src_vec->erase(it);

        if (dest_player_ != source_player_) {
            if (instance_id_ >= 0 && instance_id_ < (int)state.card_owner_map.size()) {
                state.card_owner_map[instance_id_] = dest_player_;
            }
            card.owner = dest_player_;
        } else {
             if (instance_id_ >= 0 && instance_id_ < (int)state.card_owner_map.size()) {
                 if (state.card_owner_map[instance_id_] != dest_player_) {
                     previous_owner_ = state.card_owner_map[instance_id_];
                     state.card_owner_map[instance_id_] = dest_player_;
                 }
            }
        }

        if (dest_index_ < 0 || dest_index_ >= (int)dest_vec->size()) {
            dest_vec->push_back(card);
        } else {
            dest_vec->insert(dest_vec->begin() + dest_index_, card);
        }
    }

    void TransitionCommand::invert(core::GameState& state) {
        auto* src_vec_reverse = get_zone_vector(state, dest_player_, dest_zone_);
        auto* dest_vec_reverse = get_zone_vector(state, source_player_, source_zone_);

        if (!src_vec_reverse || !dest_vec_reverse) return;

         auto it = std::find_if(src_vec_reverse->begin(), src_vec_reverse->end(),
            [this](const core::CardInstance& c) { return c.instance_id == instance_id_; });

        if (it == src_vec_reverse->end()) return;

        core::CardInstance card = *it;
        src_vec_reverse->erase(it);

        card.is_tapped = was_tapped_;
        card.is_face_down = was_face_down_;

        if (previous_owner_ != -1 && instance_id_ >= 0 && instance_id_ < (int)state.card_owner_map.size()) {
            state.card_owner_map[instance_id_] = previous_owner_;
            card.owner = previous_owner_;
        }

        if (previous_index_ < 0 || previous_index_ >= (int)dest_vec_reverse->size()) {
            dest_vec_reverse->push_back(card);
        } else {
            dest_vec_reverse->insert(dest_vec_reverse->begin() + previous_index_, card);
        }
    }

    // --- MutateCommand ---

    MutateCommand::MutateCommand(int target_id, MutationType type, int value, int duration)
        : target_id_(target_id), type_(type), value_(value), duration_(duration) {}

    MutateCommand::MutateCommand(int target_id, MutationType type, int value, const std::string& str_value)
        : target_id_(target_id), type_(type), value_(value), duration_(0), str_value_(str_value) {}

    void MutateCommand::execute(core::GameState& state) {
        core::CardInstance* card = state.get_card_instance(target_id_);

        if (card) {
             switch (type_) {
                case MutationType::TAP:
                    previous_bool_ = card->is_tapped;
                    card->is_tapped = true;
                    break;
                case MutationType::UNTAP:
                    previous_bool_ = card->is_tapped;
                    card->is_tapped = false;
                    break;
                case MutationType::POWER_MOD:
                    previous_value_ = card->power_mod;
                    card->power_mod += value_;
                    break;
                case MutationType::ADD_KEYWORD:
                    // TODO: Implement dynamic keyword addition (requires CardInstance keyword override or Modifier system)
                    // Currently CardInstance doesn't store Keywords, CardDefinition does.
                    // This requires a "Modifier" that grants keywords.
                    break;
                case MutationType::REMOVE_KEYWORD:
                    break;
                default: break;
            }
        }
    }

    void MutateCommand::invert(core::GameState& state) {
        core::CardInstance* card = state.get_card_instance(target_id_);
        if (card) {
            switch (type_) {
                case MutationType::TAP:
                case MutationType::UNTAP:
                    card->is_tapped = previous_bool_;
                    break;
                case MutationType::POWER_MOD:
                    card->power_mod = previous_value_;
                    break;
                default: break;
            }
        }
    }

    // --- FlowCommand ---

    FlowCommand::FlowCommand(FlowType type, int next_value)
        : type_(type), next_value_(next_value) {}

    void FlowCommand::execute(core::GameState& state) {
        switch (type_) {
            case FlowType::PHASE_CHANGE:
                prev_phase_ = static_cast<int>(state.current_phase);
                state.current_phase = static_cast<core::Phase>(next_value_);
                break;
            case FlowType::NEXT_TURN:
                prev_turn_ = state.turn_number;
                prev_active_player_ = state.active_player_id;
                prev_phase_ = static_cast<int>(state.current_phase);
                state.turn_number++;
                state.active_player_id = 1 - state.active_player_id;
                state.current_phase = core::Phase::START_OF_TURN;
                break;
            case FlowType::GAME_OVER:
                prev_winner_ = static_cast<int>(state.winner);
                state.winner = static_cast<core::GameResult>(next_value_);
                break;
            case FlowType::SET_ATTACK_SOURCE:
                prev_value_ = state.current_attack.source_instance_id;
                state.current_attack.source_instance_id = next_value_;
                break;
            case FlowType::SET_ATTACK_TARGET:
                prev_value_ = state.current_attack.target_instance_id;
                state.current_attack.target_instance_id = next_value_;
                break;
             case FlowType::SET_ATTACK_PLAYER:
                prev_value_ = state.current_attack.target_player;
                state.current_attack.target_player = static_cast<core::PlayerID>(next_value_);
                break;
        }
    }

    void FlowCommand::invert(core::GameState& state) {
        switch (type_) {
            case FlowType::PHASE_CHANGE:
                state.current_phase = static_cast<core::Phase>(prev_phase_);
                break;
            case FlowType::NEXT_TURN:
                state.turn_number = prev_turn_;
                state.active_player_id = prev_active_player_;
                state.current_phase = static_cast<core::Phase>(prev_phase_);
                break;
            case FlowType::GAME_OVER:
                state.winner = static_cast<core::GameResult>(prev_winner_);
                break;
            case FlowType::SET_ATTACK_SOURCE:
                state.current_attack.source_instance_id = prev_value_;
                break;
            case FlowType::SET_ATTACK_TARGET:
                state.current_attack.target_instance_id = prev_value_;
                break;
             case FlowType::SET_ATTACK_PLAYER:
                state.current_attack.target_player = static_cast<core::PlayerID>(prev_value_);
                break;
        }
    }

    // --- QueryCommand ---

    QueryCommand::QueryCommand(const std::string& query_type, const std::map<std::string, int>& params, const std::vector<int>& valid_targets)
        : query_type_(query_type), params_(params), valid_targets_(valid_targets) {}

    void QueryCommand::execute(core::GameState& state) {
        state.waiting_for_user_input = true;
        state.pending_query = core::GameState::QueryContext();
        state.pending_query->query_type = query_type_;
        state.pending_query->params = params_;
        state.pending_query->valid_target_ids = valid_targets_;
    }

    void QueryCommand::invert(core::GameState& state) {
        state.waiting_for_user_input = false;
        state.pending_query.reset();
    }

    // --- DecideCommand ---

    DecideCommand::DecideCommand(int query_id, const std::vector<int>& selected_indices)
        : query_id_(query_id), selected_indices_(selected_indices) {}

    void DecideCommand::execute(core::GameState& state) {
        state.waiting_for_user_input = false;
        state.pending_query.reset();
    }

    void DecideCommand::invert(core::GameState& state) {
        state.waiting_for_user_input = true;
    }

}
