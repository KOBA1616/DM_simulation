#include "phase_strategies.hpp"
#include "engine/systems/mechanics/mana_system.hpp"
#include "engine/utils/target_utils.hpp"
#include "core/modifiers.hpp"
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> MainPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& state = ctx.game_state;
        const auto& player = state.players[ctx.player_id];
        const auto& card_db = ctx.card_db;
        
        // 1. Play cards from hand
        for (const auto& card : player.hand) {
            if (!card_db.count(card.card_id)) continue;
            const auto& card_def = card_db.at(card.card_id);

            // Check if play is legal (cost & mana)
            if (ManaSystem::can_pay_cost(state, player, card_def, card_db)) {

                // Check Global Prohibitions (Lock Effects)
                bool prohibited = false;
                for (const auto& eff : state.passive_effects) {
                    if (eff.type == PassiveType::CANNOT_SUMMON && (card_def.type == CardType::CREATURE || card_def.type == CardType::EVOLUTION_CREATURE)) {
                         // Check if this card matches the prohibition filter
                         if (dm::engine::utils::TargetUtils::is_valid_target(card, card_def, eff.target_filter, state, eff.controller, player.id, true)) {
                             prohibited = true;
                             break;
                         }
                    }
                    if (eff.type == PassiveType::CANNOT_USE_SPELLS && card_def.type == CardType::SPELL) {
                        if (dm::engine::utils::TargetUtils::is_valid_target(card, card_def, eff.target_filter, state, eff.controller, player.id, true)) {
                             prohibited = true;
                             break;
                         }
                    }
                }
                if (prohibited) continue;

                bool is_evolution = card_def.keywords.evolution || card_def.type == CardType::EVOLUTION_CREATURE;
                bool is_neo = card_def.keywords.neo;

                // NEO Creatures can be played as NORMAL or EVOLUTION.
                // If Normal (or NEO choosing Normal):
                if (!is_evolution || is_neo) {
                    Action action;
                    action.type = PlayerIntent::PLAY_CARD;
                    action.card_id = card.card_id;
                    action.source_instance_id = card.instance_id;
                    action.target_player = player.id; // Usually self
                    // No target_instance_id for normal summon
                    actions.push_back(action);
                }

                // If Evolution (or NEO choosing Evolution):
                if (is_evolution || is_neo) {
                    // We must find valid evolution sources in Battle Zone.
                    for (const auto& source : player.battle_zone) {
                        const auto& source_def = card_db.at(source.card_id);

                        // Check validity
                        bool valid = false;

                        if (card_def.evolution_condition.has_value()) {
                            // Use the explicit evolution condition filter
                            if (dm::engine::utils::TargetUtils::is_valid_target(source, source_def, *card_def.evolution_condition, state, player.id, player.id, false)) {
                                valid = true;
                            }
                        } else {
                            // Race Match (Standard Fallback)
                            for (const auto& r : card_def.races) {
                                if (dm::engine::utils::TargetUtils::CardProperties<CardDefinition>::has_race(source_def, r)) {
                                    valid = true;
                                    break;
                                }
                            }
                        }

                        // NEO usually evolves from same Civ or Race
                        if (is_neo) valid = true;

                        if (valid) {
                            Action action;
                            action.type = PlayerIntent::PLAY_CARD;
                            action.card_id = card.card_id;
                            action.source_instance_id = card.instance_id;
                            action.target_instance_id = source.instance_id; // Evolution Source
                            action.target_player = player.id;
                            actions.push_back(action);
                        }
                    }
                }
            }
        }

        // 2. Use Abilities (e.g. Castle, Field, activated abilities on board)
        // ... (existing logic)

        // 3. Pass (End Turn)
        Action pass_action;
        pass_action.type = PlayerIntent::PASS;
        actions.push_back(pass_action);
        
        return actions;
    }
}
