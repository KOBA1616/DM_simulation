#pragma once
#include "../../core/game_state.hpp"
#include "../../core/card_json_types.hpp"
#include "../../core/card_def.hpp"
#include "card_registry.hpp"

namespace dm::engine {

    class TargetUtils {
    public:
        // Traits for extracting properties from CardDefinition or CardData
        template<typename T>
        struct CardProperties;

        // Generic validator
        template<typename CardType>
        static bool is_valid_target(const dm::core::CardInstance& card,
                                    const CardType& card_def,
                                    const dm::core::FilterDef& filter,
                                    const dm::core::GameState& game_state,
                                    dm::core::PlayerID source_controller,
                                    dm::core::PlayerID card_controller) {

            using Props = CardProperties<CardType>;

            // NOTE: Just Diver logic is handled in ActionGenerator/Attacking logic specifically for "Selection" actions.
            // is_valid_target handles general filter compliance (e.g. for counting).
            // If we put Just Diver here, counting effects (e.g. "Destroy all creatures") would skip it, which is wrong.
            // Just Diver protects against CHOOSING/TARGETING.

            // 1. Owner Check
            if (filter.owner.has_value()) {
                std::string req = filter.owner.value();
                if (req == "SELF" && card_controller != source_controller) return false;
                if (req == "OPPONENT" && card_controller == source_controller) return false;
            }

            // 3. Type Check
            if (!filter.types.empty()) {
                bool type_match = false;
                for (const auto& ft : filter.types) {
                    if (Props::match_type(card_def, ft)) { type_match = true; break; }
                }
                if (!type_match) return false;
            }

            // 4. Civilization Check
            if (!filter.civilizations.empty()) {
                 bool civ_match = false;
                 for (const auto& fc : filter.civilizations) {
                     if (Props::match_civilization(card_def, fc)) { civ_match = true; break; }
                 }
                 if (!civ_match) return false;
            }

            // 5. Race Check
            if (!filter.races.empty()) {
                bool race_match = false;
                for (const auto& fr : filter.races) {
                    if (Props::has_race(card_def, fr)) { race_match = true; break; }
                }
                if (!race_match) return false;
            }

            // 6. Cost/Power Checks
            if (filter.min_cost.has_value() && Props::get_cost(card_def) < filter.min_cost.value()) return false;
            if (filter.max_cost.has_value() && Props::get_cost(card_def) > filter.max_cost.value()) return false;
            if (filter.min_power.has_value() && Props::get_power(card_def) < filter.min_power.value()) return false;
            if (filter.max_power.has_value() && Props::get_power(card_def) > filter.max_power.value()) return false;

            // 7. State Checks
            if (filter.is_tapped.has_value() && card.is_tapped != filter.is_tapped.value()) return false;
            if (filter.is_evolution.has_value()) {
                bool is_evo = Props::is_evolution(card_def);
                if (is_evo != filter.is_evolution.value()) return false;
            }
            if (filter.is_blocker.has_value()) {
                bool is_blocker = Props::is_blocker(card_def);
                if (is_blocker != filter.is_blocker.value()) return false;
            }

            return true;
        }

        // Helper to check Just Diver status specifically
        // Returns true if the card is currently protected by Just Diver
        static bool is_protected_by_just_diver(const dm::core::CardInstance& card,
                                               const dm::core::CardDefinition& def,
                                               const dm::core::GameState& game_state,
                                               dm::core::PlayerID opponent_id) { // opponent_id is who is trying to target
             if (!def.keywords.just_diver) return false;

             dm::core::PlayerID controller = 1 - opponent_id; // Assumes 2 players

             // If source is OPPONENT (source_controller != card_controller)
             // (Implied by calling this function with opponent_id)

             // Check turn logic
             // Protected if (current_turn <= turn_played) for P0 owner
             if (controller == 0) {
                 if (game_state.turn_number == card.turn_played) return true;
             } else {
                 if (game_state.turn_number <= card.turn_played + 1) return true;
             }
             return false;
        }

    };

    // Specializations
    template<>
    struct TargetUtils::CardProperties<dm::core::CardDefinition> {
        static int get_cost(const dm::core::CardDefinition& c) { return c.cost; }
        static int get_power(const dm::core::CardDefinition& c) { return c.power; }
        static bool has_race(const dm::core::CardDefinition& c, const std::string& race) {
            for (const auto& r : c.races) if (r == race) return true;
            return false;
        }
        static bool match_civilization(const dm::core::CardDefinition& c, const std::string& civ_str) {
            using namespace dm::core;
            Civilization target_civ = Civilization::NONE;
            if (civ_str == "FIRE") target_civ = Civilization::FIRE;
            else if (civ_str == "WATER") target_civ = Civilization::WATER;
            else if (civ_str == "NATURE") target_civ = Civilization::NATURE;
            else if (civ_str == "LIGHT") target_civ = Civilization::LIGHT;
            else if (civ_str == "DARKNESS") target_civ = Civilization::DARKNESS;
            else if (civ_str == "ZERO") target_civ = Civilization::ZERO;

            return (c.civilization & target_civ) != Civilization::NONE;
        }
        static bool match_type(const dm::core::CardDefinition& c, const std::string& type_str) {
            using namespace dm::core;
            if (type_str == "CREATURE") return c.type == CardType::CREATURE || c.type == CardType::EVOLUTION_CREATURE;
            if (type_str == "SPELL") return c.type == CardType::SPELL;
            return false;
        }
        static bool is_evolution(const dm::core::CardDefinition& c) {
            return c.type == dm::core::CardType::EVOLUTION_CREATURE || c.keywords.evolution;
        }
        static bool is_blocker(const dm::core::CardDefinition& c) {
            return c.keywords.blocker;
        }
    };

    template<>
    struct TargetUtils::CardProperties<dm::core::CardData> {
        static int get_cost(const dm::core::CardData& c) { return c.cost; }
        static int get_power(const dm::core::CardData& c) { return c.power; }
        static bool has_race(const dm::core::CardData& c, const std::string& race) {
            for (const auto& r : c.races) if (r == race) return true;
            return false;
        }
        static bool match_civilization(const dm::core::CardData& c, const std::string& civ_str) {
            return c.civilization == civ_str;
        }
        static bool match_type(const dm::core::CardData& c, const std::string& type_str) {
            return c.type == type_str;
        }
        static bool is_evolution(const dm::core::CardData& c) {
            return c.type == "EVOLUTION_CREATURE";
        }
        static bool is_blocker(const dm::core::CardData& c) {
            return false; // Not easily checking keywords here
        }
    };

}
