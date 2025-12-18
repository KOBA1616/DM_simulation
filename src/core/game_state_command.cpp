#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include <iostream>
#include <algorithm>

namespace dm::engine::game_command {

    using namespace core;

    // --- TransitionCommand ---

    void TransitionCommand::execute(GameState& state) {
        // Implementation of card movement
        // We need to access zones via GameState logic or helpers
        // Since we are inside a Command which is friend/internal, we assume we can manipulate state.

        Player& p_owner = state.players[owner_id];

        // Find source card and verify zone
        std::vector<CardInstance>* src_vec = nullptr;

        // Helper to find vector
        auto get_vec = [&](PlayerID pid, Zone z) -> std::vector<CardInstance>* {
             if (pid > 1) return nullptr;
             switch(z) {
                 case Zone::HAND: return &state.players[pid].hand;
                 case Zone::MANA: return &state.players[pid].mana_zone;
                 case Zone::BATTLE: return &state.players[pid].battle_zone;
                 case Zone::GRAVEYARD: return &state.players[pid].graveyard;
                 case Zone::SHIELD: return &state.players[pid].shield_zone;
                 case Zone::DECK: return &state.players[pid].deck;
                 case Zone::BUFFER: return &state.players[pid].effect_buffer;
                 case Zone::STACK: return &state.stack_zone;
                 default: return nullptr;
             }
        };

        src_vec = get_vec(owner_id, from_zone); // Assuming owner matches location?
        // Wait, Card ownership is fixed, but location might be in opponent's zone?
        // For standard DM, cards generally stay in owner's zones except Battle Zone (ownership vs controller).
        // But get_vec uses pid. If card is in opponent's zone, we need that pid.
        // We assume command is constructed with correct "owner_id" meaning "current controller/location owner".

        if (!src_vec && from_zone != Zone::STACK) {
             // Try looking up controller
             PlayerID current_controller = state.card_owner_map.size() > (size_t)card_instance_id ? state.card_owner_map[card_instance_id] : owner_id;
             src_vec = get_vec(current_controller, from_zone);
        }

        if (!src_vec && from_zone == Zone::STACK) src_vec = &state.stack_zone;

        if (!src_vec) return;

        // Remove
        CardInstance card_obj;
        bool found = false;
        for (auto it = src_vec->begin(); it != src_vec->end(); ++it) {
            if (it->instance_id == card_instance_id) {
                card_obj = *it;
                original_index = std::distance(src_vec->begin(), it);
                src_vec->erase(it);
                found = true;
                break;
            }
        }

        if (!found) return;

        // Add
        std::vector<CardInstance>* dst_vec = get_vec(owner_id, to_zone); // Destination is usually owner's zone
        if (!dst_vec && to_zone == Zone::STACK) dst_vec = &state.stack_zone;

        if (dst_vec) {
            if (destination_index == -1 || destination_index >= (int)dst_vec->size()) {
                dst_vec->push_back(card_obj);
            } else {
                dst_vec->insert(dst_vec->begin() + destination_index, card_obj);
            }
        }

        // Update Owner Map if needed? Usually only if changing controllers.
        // TransitionCommand assumes owner_id is the DESTINATION owner.
        if (state.card_owner_map.size() <= (size_t)card_instance_id) {
             state.card_owner_map.resize(card_instance_id + 1, 255);
        }
        state.card_owner_map[card_instance_id] = owner_id;
    }

    void TransitionCommand::invert(GameState& state) {
        // Swap to/from
        // Use original_index to restore position
        // This requires careful implementation, skipping for MVP/Stub
    }

    // --- MutateCommand ---

    void MutateCommand::execute(GameState& state) {
        if (target_instance_id == -1) {
            // Global mutations
            if (mutation_type == MutationType::ADD_PASSIVE_EFFECT && passive_effect.has_value()) {
                state.passive_effects.push_back(passive_effect.value());
            } else if (mutation_type == MutationType::ADD_COST_MODIFIER && cost_modifier.has_value()) {
                state.active_modifiers.push_back(cost_modifier.value());
            }
            return;
        }

        CardInstance* card = state.get_card_instance(target_instance_id);
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
                previous_int_value = card->power_mod;
                card->power_mod += int_value;
                break;
            // Keywords etc...
            default: break;
        }
    }

    void MutateCommand::invert(GameState& state) {
        // ...
    }

    // --- AttachCommand (New) ---

    void AttachCommand::execute(GameState& state) {
        CardInstance* base_card = state.get_card_instance(target_base_card_id);
        if (!base_card) return;

        // 1. Remove source card from its zone
        Player& owner_p = state.players[state.active_player_id]; // Assume active player plays it
        std::vector<CardInstance>* src_vec = nullptr;
        if (source_zone == Zone::HAND) src_vec = &owner_p.hand;
        else if (source_zone == Zone::BATTLE) src_vec = &owner_p.battle_zone; // Unlikely but possible

        if (!src_vec) return;

        auto it = std::find_if(src_vec->begin(), src_vec->end(), [&](const CardInstance& c){ return c.instance_id == card_to_attach_id; });
        if (it == src_vec->end()) return;

        CardInstance attach_card = *it;
        original_zone_index = std::distance(src_vec->begin(), it);
        src_vec->erase(it);

        // 2. Perform Evolution (Stack on Top)
        // Logic: The 'attach_card' (Evolution Creature) becomes the main card in the Battle Zone.
        // The 'base_card' is moved INTO 'attach_card.underlying_cards'.
        // AND 'base_card.underlying_cards' (if any) are also moved to 'attach_card.underlying_cards'.
        // State (Tapped/Sick) is inherited from Base.

        // Inherit state
        attach_card.is_tapped = base_card->is_tapped;
        attach_card.summoning_sickness = base_card->summoning_sickness;
        attach_card.turn_played = base_card->turn_played;

        // Move base and its underlying to new card
        attach_card.underlying_cards.push_back(*base_card); // Push base
        attach_card.underlying_cards.insert(attach_card.underlying_cards.end(),
                                            base_card->underlying_cards.begin(),
                                            base_card->underlying_cards.end());
        base_card->underlying_cards.clear(); // Clear form base (though base is effectively gone)

        // Replace base in Battle Zone with attach_card
        // We need to find base_card in its zone (Battle Zone)
        PlayerID base_owner_id = state.card_owner_map[target_base_card_id];
        Player& base_owner = state.players[base_owner_id];

        auto base_it = std::find_if(base_owner.battle_zone.begin(), base_owner.battle_zone.end(),
                                    [&](const CardInstance& c){ return c.instance_id == target_base_card_id; });

        if (base_it != base_owner.battle_zone.end()) {
             *base_it = attach_card; // Overwrite in place
             state.card_owner_map[attach_card.instance_id] = base_owner_id; // Update map for new card
             // Note: base_card's ID is now inside, so map should still point to owner?
             // Yes, owner logic handles "underlying" by searching parent.
             // But get_card_instance searches underlying.
        }
    }

    void AttachCommand::invert(GameState& state) {
        // TODO
    }

    // --- FlowCommand ---
    void FlowCommand::execute(GameState& state) { /*...*/ }
    void FlowCommand::invert(GameState& state) { /*...*/ }

    // --- Query/Decide ---
    void QueryCommand::execute(GameState& state) {
        state.waiting_for_user_input = true;
        state.pending_query = GameState::QueryContext{0, query_type, params, valid_targets, {}};
    }
    void QueryCommand::invert(GameState& state) { /*...*/ }

    void DecideCommand::execute(GameState& state) {
        state.waiting_for_user_input = false;
        state.pending_query.reset();
        // Logic handled by resume mechanism, this command records the decision
    }
    void DecideCommand::invert(GameState& state) { /*...*/ }

    // --- DeclareReaction ---
    void DeclareReactionCommand::execute(GameState& state) {
        // Logic usually handled by ReactionSystem resolving this command
        if (state.status == GameState::Status::WAITING_FOR_REACTION) {
             state.status = GameState::Status::PLAYING;
             if (!state.reaction_stack.empty()) {
                 state.reaction_stack.pop_back(); // Close window
             }
        }
    }
    void DeclareReactionCommand::invert(GameState& state) { /*...*/ }

    // --- Stat/Result ---
    void StatCommand::execute(GameState& state) { /*...*/ }
    void StatCommand::invert(GameState& state) { /*...*/ }
    void GameResultCommand::execute(GameState& state) { state.winner = result; state.is_game_over = true; }
    void GameResultCommand::invert(GameState& state) { state.winner = previous_result; state.is_game_over = false; }
}

namespace dm::core {
    void GameState::execute_command(std::shared_ptr<dm::engine::game_command::GameCommand> cmd) {
        if (!cmd) return;
        cmd->execute(*this);
        command_history.push_back(cmd);
    }
}
