#include "phase_strategies.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "core/modifiers.hpp"
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    void MainPhaseStrategy::generate(const GameState& state, const Player& player, std::vector<Action>& actions, const std::map<CardID, CardDefinition>& card_db) {
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
                         if (TargetUtils::is_valid_target(card, card_def, eff.target_filter, state, eff.controller, player.id, true)) {
                             prohibited = true;
                             break;
                         }
                    }
                    if (eff.type == PassiveType::CANNOT_USE_SPELLS && card_def.type == CardType::SPELL) {
                        if (TargetUtils::is_valid_target(card, card_def, eff.target_filter, state, eff.controller, player.id, true)) {
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
                    // Evolution filter is usually implicit in the card definition?
                    // Currently, CardDefinition doesn't store "Evolution Condition" (Race/Civ).
                    // We need to support it.
                    // Assumption: `races` field implies race requirement? Or filter_ids?
                    // For now, let's assume ANY creature matching Race/Civ is valid if logic exists.
                    // Or simply: If it IS evolution, we generate actions for each valid target.

                    // Filter Logic:
                    // If card_def has `evolution_condition` (not yet in struct), we use it.
                    // Or we check `races` for race evolution.

                    // Let's iterate all creatures and check compatibility.
                    // Simple logic: Same Race or Civilization?
                    // Standard DM: Evolution is usually "Race Evolution".
                    // Let's check races.

                    for (const auto& source : player.battle_zone) {
                        const auto& source_def = card_db.at(source.card_id);

                        // Check validity
                        bool valid = false;

                        // 1. Race Match (Standard)
                        // If the evolution creature specifies a race to evolve from...
                        // But CardDefinition doesn't explicitly say "Evolves from X".
                        // It just has "Races".
                        // For MVP, we assume "Evolves from SAME Race" or "Matches Civilization"?
                        // Let's use `TargetUtils` if we had a filter.
                        // Without explicit filter, we'll be permissive or strict based on race intersection.

                        for (const auto& r : card_def.races) {
                            if (TargetUtils::CardProperties<CardDefinition>::has_race(source_def, r)) {
                                valid = true;
                                break;
                            }
                        }

                        // NEO usually evolves from same Civ or Race?
                        // NEO: "Put on one of your creatures". Usually generic.
                        // Let's assume NEO can evolve on ANY creature for now (or same Civ).
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

        // 3. Pass (End Turn) - Always legal in Main Phase if stack is empty?
        // Actually Main Phase -> Attack Phase.
        // If we want to end turn, we pass to Attack Phase, then Pass in Attack Phase.
        // But PlayerIntent::PASS usually means "Next Phase".
        Action pass_action;
        pass_action.type = PlayerIntent::PASS;
        actions.push_back(pass_action);
    }
}
