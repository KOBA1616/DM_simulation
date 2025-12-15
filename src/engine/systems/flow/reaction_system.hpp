#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/constants.hpp"
#include "engine/systems/card/target_utils.hpp"
#include <map>
#include <string>

namespace dm::engine {

    class ReactionSystem {
    public:
        // Checks for reactions and pushes a REACTION_WINDOW pending effect if any are available.
        // Returns true if a window was opened.
        static bool check_and_open_window(
            dm::core::GameState& game_state,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            const std::string& trigger_event,
            dm::core::PlayerID reaction_player_id
        ) {
            bool has_reaction = false;
            auto& player = game_state.players[reaction_player_id];

            int attacker_id = game_state.current_attack.source_instance_id;
            bool attacker_exists = false;
            if (attacker_id != -1) {
                // Check if attacker still in battle zone
                dm::core::PlayerID att_player_id = game_state.active_player_id; // Attacker is active
                auto& att_player = game_state.players[att_player_id];
                 auto it = std::find_if(att_player.battle_zone.begin(), att_player.battle_zone.end(), [&](const dm::core::CardInstance& c){ return c.instance_id == attacker_id; });
                 if (it != att_player.battle_zone.end()) {
                     attacker_exists = true;
                 }
            }

            // 1. Scan Hand for Ninja Strike / Strike Back
            for (const auto& card : player.hand) {
                if (!card_db.count(card.card_id)) continue;
                const auto& def = card_db.at(card.card_id);

                for (const auto& reaction : def.reaction_abilities) {
                    // Check Event Type Match
                    bool event_match = (reaction.condition.trigger_event == trigger_event);
                    if (!event_match) {
                        if (reaction.condition.trigger_event == "ON_BLOCK_OR_ATTACK") {
                            if (trigger_event == "ON_ATTACK" || trigger_event == "ON_BLOCK") {
                                event_match = true;
                            }
                        }
                    }
                    if (!event_match) continue;

                    // Check zone if specified
                    if (!reaction.zone.empty()) {
                         if (reaction.zone == "HAND") {
                             if (trigger_event == "ON_ATTACK" || trigger_event == "ON_BLOCK") {
                                 if (!attacker_exists) continue;
                             }
                         } else {
                             // If reaction zone is NOT HAND, skip
                             continue;
                         }
                    }

                    // Check other conditions (Mana Count, Civ Match)
                    if (!check_condition(game_state, reaction, card, reaction_player_id, card_db)) continue;

                    has_reaction = true;
                    break;
                }
                if (has_reaction) break;
            }

            if (has_reaction) {
                dm::core::PendingEffect effect(dm::core::EffectType::REACTION_WINDOW, -1, reaction_player_id);
                effect.resolve_type = dm::core::ResolveType::EFFECT_RESOLUTION; // Handled by ActionGenerator
                effect.optional = true; // Player can choose PASS

                // Fix: ReactionContext is now in dm::core namespace, not nested in PendingEffect
                dm::core::ReactionContext context;
                context.trigger_event = trigger_event;
                // Add more context if needed (attacker ID etc)
                effect.reaction_context = context;

                game_state.pending_effects.push_back(effect);

                // Spec 6.2.1: Update GameState flag
                game_state.waiting_for_reaction = true;
                game_state.reaction_context = context;

                return true;
            }

            return false;
        }

    public:
        static bool check_condition(
            const dm::core::GameState& game_state,
            const dm::core::ReactionAbility& reaction,
            const dm::core::CardInstance& card_instance,
            dm::core::PlayerID player_id,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            const std::string& event_type = ""
        ) {
            // Event Type Match (if provided)
            if (!event_type.empty()) {
                bool event_match = (reaction.condition.trigger_event == event_type);
                if (!event_match) {
                    if (reaction.condition.trigger_event == "ON_BLOCK_OR_ATTACK") {
                        if (event_type == "ON_ATTACK" || event_type == "ON_BLOCK") {
                            event_match = true;
                        }
                    }
                }
                if (!event_match) return false;
            }

            // Civilization Match
            if (reaction.condition.civilization_match) {
                const auto& player = game_state.players[player_id];
                // Check if mana zone has matching civilization
                const auto& def = card_db.at(card_instance.card_id);
                bool match_found = false;

                // Simple Civ Check - Iterate all civilizations
                for (const auto& mana_card : player.mana_zone) {
                    if (!card_db.count(mana_card.card_id)) continue;
                    const auto& m_def = card_db.at(mana_card.card_id);

                    // Cross check enum lists
                    for (const auto& req_civ : def.civilizations) {
                         for (const auto& man_civ : m_def.civilizations) {
                             if (req_civ == man_civ) {
                                 match_found = true;
                                 break;
                             }
                         }
                         if (match_found) break;
                    }
                    if (match_found) break;
                }
                if (!match_found) return false;
            }

            // Mana Count Min
            if (reaction.condition.mana_count_min > 0) {
                 const auto& player = game_state.players[player_id];
                 // Basic count, assumes no tapped/untapped logic for raw count
                 if (player.mana_zone.size() < (size_t)reaction.condition.mana_count_min) {
                     return false;
                 }
            }

            return true;
        }
    };
}
