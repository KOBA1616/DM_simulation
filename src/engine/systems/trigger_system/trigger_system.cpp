#include "trigger_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "core/game_event.hpp"
#include "engine/game_command/commands.hpp"
#include <iostream>

namespace dm::engine::systems {

    using namespace core;

    void TriggerSystem::resolve_trigger(GameState& game_state, TriggerType trigger, int source_instance_id, const std::map<CardID, CardDefinition>& card_db) {
        CardInstance* instance = game_state.get_card_instance(source_instance_id);
        if (!instance) {
            return;
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

        PlayerID controller = get_controller(game_state, source_instance_id);

        for (const auto& effect : active_effects) {
            if (effect.trigger == trigger) {
                PendingEffect pending(EffectType::TRIGGER_ABILITY, source_instance_id, controller);
                pending.resolve_type = ResolveType::EFFECT_RESOLUTION;
                pending.effect_def = effect;
                pending.optional = true;
                pending.chain_depth = game_state.turn_stats.current_chain_depth + 1;

                auto cmd = std::make_unique<dm::engine::game_command::MutateCommand>(-1, dm::engine::game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT);
                cmd->pending_effect = pending;
                game_state.execute_command(std::move(cmd));
            }
        }
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
