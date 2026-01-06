#include "engine/game_command/commands.hpp"
#include "core/game_state.hpp"
#include "engine/systems/game_logic_system.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine::game_command {

    using namespace core;

    // --- TransitionCommand ---
    void TransitionCommand::execute(GameState& state) {
        if (owner_id >= state.players.size()) return;
        Player& p_owner = state.players[owner_id];

        auto remove_from_zone = [&](std::vector<CardInstance>& zone) -> std::optional<CardInstance> {
            auto it = std::find_if(zone.begin(), zone.end(),
                [&](const CardInstance& c){ return c.instance_id == card_instance_id; });
            if (it != zone.end()) {
                original_index = std::distance(zone.begin(), it);
                CardInstance c = *it;
                zone.erase(it);
                return c;
            }
            return std::nullopt;
        };

        std::optional<CardInstance> card_opt;

        switch (from_zone) {
            case Zone::HAND: card_opt = remove_from_zone(p_owner.hand); break;
            case Zone::MANA: card_opt = remove_from_zone(p_owner.mana_zone); break;
            case Zone::BATTLE: card_opt = remove_from_zone(p_owner.battle_zone); break;
            case Zone::SHIELD: card_opt = remove_from_zone(p_owner.shield_zone); break;
            case Zone::GRAVEYARD: card_opt = remove_from_zone(p_owner.graveyard); break;
            case Zone::DECK: card_opt = remove_from_zone(p_owner.deck); break;
            case Zone::BUFFER: card_opt = remove_from_zone(p_owner.effect_buffer); break;
            case Zone::STACK: card_opt = remove_from_zone(p_owner.stack); break;
            default: break;
        }

        if (card_opt) {
            CardInstance& card = card_opt.value();
            if (to_zone == Zone::HAND || to_zone == Zone::DECK) {
                card.is_tapped = false;
                card.summoning_sickness = true;
                card.underlying_cards.clear();
            }
            if (to_zone == Zone::BATTLE) {
                card.turn_played = state.turn_number;
                card.summoning_sickness = true;
            }

            std::vector<CardInstance>* dest_zone_ptr = nullptr;
            switch (to_zone) {
                case Zone::HAND: dest_zone_ptr = &p_owner.hand; break;
                case Zone::MANA: dest_zone_ptr = &p_owner.mana_zone; break;
                case Zone::BATTLE: dest_zone_ptr = &p_owner.battle_zone; break;
                case Zone::SHIELD: dest_zone_ptr = &p_owner.shield_zone; break;
                case Zone::GRAVEYARD: dest_zone_ptr = &p_owner.graveyard; break;
                case Zone::DECK: dest_zone_ptr = &p_owner.deck; break;
                case Zone::BUFFER: dest_zone_ptr = &p_owner.effect_buffer; break;
                case Zone::STACK: dest_zone_ptr = &p_owner.stack; break;
                default: break;
            }

            if (dest_zone_ptr) {
                if (destination_index >= 0 && destination_index <= (int)dest_zone_ptr->size()) {
                    dest_zone_ptr->insert(dest_zone_ptr->begin() + destination_index, card);
                } else {
                    dest_zone_ptr->push_back(card);
                }
            }
        }
    }

    void TransitionCommand::invert(GameState& state) {
        TransitionCommand reverse_cmd(card_instance_id, to_zone, from_zone, owner_id, original_index);
        reverse_cmd.execute(state);
    }

    // --- MutateCommand ---
    void MutateCommand::execute(GameState& state) {
        CardInstance* card = state.get_card_instance(target_instance_id);

        if (target_instance_id == -1) {
            if (mutation_type == MutationType::ADD_PASSIVE_EFFECT && passive_effect.has_value()) {
                state.passive_effects.push_back(passive_effect.value());
            } else if (mutation_type == MutationType::ADD_COST_MODIFIER && cost_modifier.has_value()) {
                state.active_modifiers.push_back(cost_modifier.value());
            } else if (mutation_type == MutationType::ADD_PENDING_EFFECT && pending_effect.has_value()) {
                state.pending_effects.push_back(pending_effect.value());
            }
            return;
        }

        if (!card) return;

        switch (mutation_type) {
            case MutationType::TAP:
                previous_bool_value = card->is_tapped;
                card->is_tapped = true;
                break;
            case MutationType::UNTAP:
                previous_bool_value = card->is_tapped;
                card->is_tapped = false;
                break;
            case MutationType::POWER_MOD:
                previous_int_value = card->power_modifier;
                card->power_modifier += int_value;
                break;
            case MutationType::SET_SUMMONING_SICKNESS:
                previous_bool_value = card->summoning_sickness;
                card->summoning_sickness = (int_value != 0);
                break;
            default: break;
        }
    }

    void MutateCommand::invert(GameState& state) {
        CardInstance* card = state.get_card_instance(target_instance_id);
        if (target_instance_id == -1) {
             if (mutation_type == MutationType::ADD_PASSIVE_EFFECT) state.passive_effects.pop_back();
             if (mutation_type == MutationType::ADD_COST_MODIFIER) state.active_modifiers.pop_back();
             if (mutation_type == MutationType::ADD_PENDING_EFFECT) state.pending_effects.pop_back();
             return;
        }
        if (!card) return;

        switch (mutation_type) {
            case MutationType::TAP:
            case MutationType::UNTAP:
                card->is_tapped = previous_bool_value;
                break;
             case MutationType::POWER_MOD:
                card->power_modifier = previous_int_value;
                break;
             case MutationType::SET_SUMMONING_SICKNESS:
                card->summoning_sickness = previous_bool_value;
                break;
            default: break;
        }
    }

    // --- AttachCommand (Evolution/Cross) ---
    void AttachCommand::execute(GameState& state) {
        CardInstance* source_card_ptr = state.get_card_instance(card_to_attach_id);
        if (!source_card_ptr) return;

        CardID new_card_id = source_card_ptr->card_id;
        PlayerID source_owner = source_card_ptr->owner;

        CardInstance* base_card = state.get_card_instance(target_base_card_id);
        if (!base_card) return;

        target_was_tapped = base_card->is_tapped;
        target_was_sick = base_card->summoning_sickness;
        original_zone = source_zone;

        Player& owner = state.players[source_owner];
        std::vector<CardInstance>* src_vec = nullptr;
        if (source_zone == Zone::HAND) src_vec = &owner.hand;
        else if (source_zone == Zone::MANA) src_vec = &owner.mana_zone;
        else if (source_zone == Zone::GRAVEYARD) src_vec = &owner.graveyard;

        if (src_vec) {
            auto it = std::find_if(src_vec->begin(), src_vec->end(),
                [&](const CardInstance& c){ return c.instance_id == card_to_attach_id; });
            if (it != src_vec->end()) {
                original_zone_index = std::distance(src_vec->begin(), it);
                src_vec->erase(it);
            }
        }

        CardInstance underlying_part;
        underlying_part.card_id = base_card->card_id;
        underlying_part.owner = base_card->owner;
        underlying_part.is_tapped = false;

        base_card->underlying_cards.insert(base_card->underlying_cards.begin(), underlying_part);
        base_card->card_id = new_card_id;
        base_card->summoning_sickness = false;
    }

    void AttachCommand::invert(GameState& state) {
        CardInstance* base_card = state.get_card_instance(target_base_card_id);
        if (!base_card || base_card->underlying_cards.empty()) return;

        CardInstance old_top = base_card->underlying_cards[0];
        CardID current_evo_id = base_card->card_id;

        base_card->card_id = old_top.card_id;
        base_card->underlying_cards.erase(base_card->underlying_cards.begin());

        base_card->is_tapped = target_was_tapped;
        base_card->summoning_sickness = target_was_sick;

        CardInstance source_card_instance;
        source_card_instance.instance_id = card_to_attach_id;
        source_card_instance.card_id = current_evo_id;
        source_card_instance.owner = base_card->owner;

        Player& p = state.players[source_card_instance.owner];
        if (original_zone == Zone::HAND) {
            if (original_zone_index >= 0 && original_zone_index <= (int)p.hand.size())
                p.hand.insert(p.hand.begin() + original_zone_index, source_card_instance);
            else p.hand.push_back(source_card_instance);
        }
        else if (original_zone == Zone::MANA) {
             if (original_zone_index >= 0 && original_zone_index <= (int)p.mana_zone.size())
                p.mana_zone.insert(p.mana_zone.begin() + original_zone_index, source_card_instance);
            else p.mana_zone.push_back(source_card_instance);
        }
        else if (original_zone == Zone::GRAVEYARD) {
             if (original_zone_index >= 0 && original_zone_index <= (int)p.graveyard.size())
                p.graveyard.insert(p.graveyard.begin() + original_zone_index, source_card_instance);
            else p.graveyard.push_back(source_card_instance);
        }
    }

    // --- FlowCommand ---
    void FlowCommand::execute(GameState& state) {
        switch (flow_type) {
            case FlowType::PHASE_CHANGE:
                previous_value = static_cast<int>(state.current_phase);
                state.current_phase = static_cast<Phase>(new_value);
                break;
            case FlowType::TURN_CHANGE:
                previous_value = state.turn_number;
                state.turn_number = new_value;
                break;
            case FlowType::SET_ACTIVE_PLAYER:
                previous_value = state.active_player_id;
                state.active_player_id = static_cast<PlayerID>(new_value);
                break;
            case FlowType::RESET_TURN_STATS:
                previous_turn_stats = state.turn_stats;
                state.turn_stats = TurnStats{};
                break;
            case FlowType::CLEANUP_STEP:
                {
                     auto& mods = state.active_modifiers;
                     for (const auto& m : mods) {
                         if (m.turns_remaining == 1) removed_modifiers.push_back(m);
                     }
                     mods.erase(std::remove_if(mods.begin(), mods.end(), [](CostModifier& m) {
                         if (m.turns_remaining > 0) m.turns_remaining--;
                         return m.turns_remaining == 0;
                     }), mods.end());

                     auto& passives = state.passive_effects;
                     for (const auto& p : passives) {
                         if (p.turns_remaining == 1) removed_passives.push_back(p);
                     }
                     passives.erase(std::remove_if(passives.begin(), passives.end(), [](PassiveEffect& p) {
                         if (p.turns_remaining > 0) p.turns_remaining--;
                         return p.turns_remaining == 0;
                     }), passives.end());
                }
                break;
            default: break;
        }
    }
    void FlowCommand::invert(GameState& state) {
        switch (flow_type) {
            case FlowType::PHASE_CHANGE:
                state.current_phase = static_cast<Phase>(previous_value);
                break;
            case FlowType::TURN_CHANGE:
                state.turn_number = previous_value;
                break;
            case FlowType::SET_ACTIVE_PLAYER:
                state.active_player_id = static_cast<PlayerID>(previous_value);
                break;
            case FlowType::RESET_TURN_STATS:
                state.turn_stats = previous_turn_stats;
                break;
            case FlowType::CLEANUP_STEP:
                // Restore removed items (Note: Survivors have decremented counters, which is not reverted here yet)
                state.active_modifiers.insert(state.active_modifiers.end(), removed_modifiers.begin(), removed_modifiers.end());
                state.passive_effects.insert(state.passive_effects.end(), removed_passives.begin(), removed_passives.end());
                break;
            default: break;
        }
    }

    // --- QueryCommand ---
    void QueryCommand::execute(GameState& state) {
        state.waiting_for_user_input = true;
        state.pending_query.query_type = query_type;
        state.pending_query.valid_targets = valid_targets;
        state.pending_query.params = params;
        state.pending_query.options = options;
    }
    void QueryCommand::invert(GameState& state) {
        state.waiting_for_user_input = false;
        state.pending_query = GameState::QueryContext{};
    }

    // --- DecideCommand ---
    void DecideCommand::execute(GameState& state) {
        state.waiting_for_user_input = false;
    }
    void DecideCommand::invert(GameState& state) {
        state.waiting_for_user_input = true;
        if (previous_query.has_value()) state.pending_query = previous_query.value();
    }

    // --- DeclareReactionCommand ---
    void DeclareReactionCommand::execute(GameState& state) {
    }
    void DeclareReactionCommand::invert(GameState& state) {
    }

    // --- StatCommand ---
    void StatCommand::execute(GameState& state) {
    }
    void StatCommand::invert(GameState& state) {
    }

    // --- GameResultCommand ---
    void GameResultCommand::execute(GameState& state) {
        state.winner = result;
        state.game_over = true;
    }
    void GameResultCommand::invert(GameState& state) {
        state.winner = previous_result;
        state.game_over = false;
    }

    // --- ShuffleCommand ---
    void ShuffleCommand::execute(GameState& state) {
        if (player_id >= state.players.size()) return;
        Player& p = state.players[player_id];

        // Store original order for undo
        original_deck_order.clear();
        original_deck_order.reserve(p.deck.size());
        for (const auto& card : p.deck) {
            original_deck_order.push_back(card.instance_id);
        }

        // Shuffle
        std::shuffle(p.deck.begin(), p.deck.end(), state.rng);
    }

    void ShuffleCommand::invert(GameState& state) {
        if (player_id >= state.players.size()) return;
        Player& p = state.players[player_id];

        std::vector<CardInstance> restored_deck;
        restored_deck.reserve(original_deck_order.size());

        std::map<int, CardInstance> temp_map;
        for (const auto& c : p.deck) {
            temp_map.insert({c.instance_id, c});
        }

        for (int id : original_deck_order) {
            if (temp_map.count(id)) {
                restored_deck.push_back(temp_map.at(id));
            }
        }

        p.deck = std::move(restored_deck);
    }

}
