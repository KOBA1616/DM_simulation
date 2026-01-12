#include "trigger_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "core/game_event.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include <iostream>

namespace dm::engine::systems {

    using namespace core;

    std::vector<EffectDef> TriggerSystem::get_trigger_effects(GameState& game_state, TriggerType trigger, int source_instance_id, const std::map<CardID, CardDefinition>& card_db) {
        std::vector<EffectDef> matching_effects;
        CardInstance* instance = game_state.get_card_instance(source_instance_id);
        if (!instance) {
            return matching_effects;
        }

        std::vector<EffectDef> active_effects;

        const CardDefinition* def_ptr = nullptr;
        if (card_db.count(instance->card_id)) {
            def_ptr = &card_db.at(instance->card_id);
        } else if (CardRegistry::get_all_definitions().count(instance->card_id)) {
            def_ptr = &CardRegistry::get_all_definitions().at(instance->card_id);
        }

        if (def_ptr) {
            const auto& data = *def_ptr;
            active_effects.insert(active_effects.end(), data.effects.begin(), data.effects.end());
            active_effects.insert(active_effects.end(), data.metamorph_abilities.begin(), data.metamorph_abilities.end());
        }

        // Handle underlying cards (e.g., evolution sources granting effects)
        for (const auto& under : instance->underlying_cards) {
            if (card_db.count(under.card_id)) {
                const auto& under_data = card_db.at(under.card_id);
                active_effects.insert(active_effects.end(), under_data.metamorph_abilities.begin(), under_data.metamorph_abilities.end());
            } else if (CardRegistry::get_all_definitions().count(under.card_id)) {
                const auto& under_data = CardRegistry::get_all_definitions().at(under.card_id);
                active_effects.insert(active_effects.end(), under_data.metamorph_abilities.begin(), under_data.metamorph_abilities.end());
            }
        }

        for (const auto& effect : active_effects) {
            if (effect.trigger == trigger) {
                // Ensure legacy behavior: only return if scope is NONE or SELF
                if (effect.trigger_scope == TargetScope::NONE || effect.trigger_scope == TargetScope::SELF) {
                    matching_effects.push_back(effect);
                }
            }
        }
        return matching_effects;
    }

    void TriggerSystem::resolve_trigger(GameState& game_state, TriggerType trigger, int source_instance_id, const std::map<CardID, CardDefinition>& card_db) {
        // 1. Resolve Standard Triggers (Scope: NONE/SELF)
        auto effects = get_trigger_effects(game_state, trigger, source_instance_id, card_db);
        PlayerID source_controller = get_controller(game_state, source_instance_id);

        for (const auto& effect : effects) {
            PendingEffect pending(EffectType::TRIGGER_ABILITY, source_instance_id, source_controller);
            pending.resolve_type = ResolveType::EFFECT_RESOLUTION;
            pending.effect_def = effect;
            pending.optional = true;
            pending.chain_depth = game_state.turn_stats.current_chain_depth + 1;

            add_pending_effect(game_state, pending);
        }

        // 2. Resolve Global Triggers (Scope: OBSERVER based)
        // We iterate through all cards in the Battle Zone of both players to find observers.

        // Get the source card instance and definition for filtering
        const CardInstance* source_card = game_state.get_card_instance(source_instance_id);
        const CardDefinition* source_def = nullptr;
        if (source_card) {
             if (card_db.count(source_card->card_id)) {
                source_def = &card_db.at(source_card->card_id);
            } else if (CardRegistry::get_all_definitions().count(source_card->card_id)) {
                source_def = &CardRegistry::get_all_definitions().at(source_card->card_id);
            }
        }

        // Only process if we have source info (some triggers might be non-card based, but resolve_trigger usually implies a source card)
        // If source_card is null (e.g. destroyed already?), we might need to rely on history or context?
        // For ON_DESTROY, the card is in Graveyard. get_card_instance uses card_owner_map which helps locate it.
        // But get_card_instance(id) might fail if it's completely gone? No, instance_id should be stable.

        if (!source_card || !source_def) return;

        // Iterate all potential observers
        // Optimization: only Battle Zone usually has triggers active.
        // Some cards in Hand (Ninja Strike) or Mana (Mana Armed?) might have triggers, but typically triggers are Battle Zone based.
        // The requirement mentions "Creature enters", "Creature attacks" etc.

        std::vector<int> potential_observers;
        for (const auto& player : game_state.players) {
            for (const auto& card : player.battle_zone) {
                // Don't trigger itself again as a global trigger if we already did (Scope check below handles this)
                potential_observers.push_back(card.instance_id);
            }
            // Add other zones if necessary? For now, standard triggers are permanent-based.
        }

        for (int obs_id : potential_observers) {
            const CardInstance* obs_card = game_state.get_card_instance(obs_id);
            if (!obs_card) continue;

            // Get Observer Definition
            const CardDefinition* obs_def = nullptr;
             if (card_db.count(obs_card->card_id)) {
                obs_def = &card_db.at(obs_card->card_id);
            } else if (CardRegistry::get_all_definitions().count(obs_card->card_id)) {
                obs_def = &CardRegistry::get_all_definitions().at(obs_card->card_id);
            }
            if (!obs_def) continue;

            // Collect triggers from observer
            std::vector<EffectDef> obs_effects = obs_def->effects;
            // Also add underlying?
            // Simplified for now.

            for (const auto& effect : obs_effects) {
                if (effect.trigger != trigger) continue;
                if (effect.trigger_scope == TargetScope::NONE || effect.trigger_scope == TargetScope::SELF) continue; // Handled in Pass 1 for self, ignored for others

                PlayerID obs_controller = get_controller(game_state, obs_id);
                PlayerID src_controller = source_controller; // from above

                // Check Scope
                bool scope_match = false;
                if (effect.trigger_scope == TargetScope::ALL_PLAYERS) {
                    scope_match = true;
                } else if (effect.trigger_scope == TargetScope::PLAYER_SELF) {
                    if (obs_controller == src_controller) scope_match = true;
                } else if (effect.trigger_scope == TargetScope::PLAYER_OPPONENT) {
                    if (obs_controller != src_controller) scope_match = true;
                }

                if (!scope_match) continue;

                // Check Filter
                // The filter applies to the SOURCE card.
                // We use TargetUtils to validate the source card against the filter.
                // We use "ignore_passives=true" to avoid complex loops during trigger checking if possible.
                // execution_context is null.
                bool filter_match = true;

                // We assume FilterDef is present if trigger_filter is set (it's a struct, so always present, but maybe empty means 'match all'?)
                // TargetUtils::is_valid_target handles empty filter as 'match all'?
                // Wait, TargetUtils checks "if (filter.types.empty())" etc. So yes.
                // Note: The default constructor of FilterDef creates empty vectors/optionals.

                // One edge case: is_valid_target checks 'owner' field in FilterDef.
                // If filter.owner is "SELF", it checks against card_controller vs source_controller (args).
                // Our usage of is_valid_target:
                // card = source_card
                // card_def = source_def
                // filter = effect.trigger_filter
                // source_controller (in is_valid_target arg) = usually who is "selecting", here it's the observer.
                // card_controller = src_controller.

                // Wait, TargetUtils args: (card, card_def, filter, game_state, source_controller, card_controller)
                // source_controller: The player "evaluating" or "targeting". So the Observer's controller.
                // card_controller: The owner of the card being checked.

                filter_match = TargetUtils::is_valid_target(*source_card, *source_def, effect.trigger_filter, game_state, obs_controller, src_controller, true);

                if (filter_match) {
                     // Queue Effect
                     // Note: Source of the effect is the Observer card (obs_id).
                     PendingEffect pending(EffectType::TRIGGER_ABILITY, obs_id, obs_controller);
                     pending.resolve_type = ResolveType::EFFECT_RESOLUTION;
                     pending.effect_def = effect;
                     pending.optional = true;
                     pending.chain_depth = game_state.turn_stats.current_chain_depth + 1;

                     add_pending_effect(game_state, pending);
                }
            }
        }
    }

    void TriggerSystem::add_pending_effect(core::GameState& game_state, const core::PendingEffect& pending_effect) {
        auto cmd = std::make_unique<dm::engine::game_command::MutateCommand>(-1, dm::engine::game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT);
        cmd->pending_effect = pending_effect;
        game_state.execute_command(std::move(cmd));
    }

    PlayerID TriggerSystem::get_controller(const GameState& game_state, int instance_id) {
        const CardInstance* card = game_state.get_card_instance(instance_id);
        if (card) {
            return card->owner;
        }

        if (instance_id >= 0 && instance_id < (int)game_state.card_owner_map.size()) {
            return game_state.card_owner_map[instance_id];
        }
        return game_state.active_player_id;
    }

}
