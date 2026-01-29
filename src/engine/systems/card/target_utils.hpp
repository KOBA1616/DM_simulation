#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "core/card_def.hpp"
#include "card_registry.hpp"
#include <algorithm>

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
                                    dm::core::PlayerID card_controller,
                                    bool ignore_passives = false,
                                    const std::map<std::string, int>* execution_context = nullptr) {

            using Props = CardProperties<CardType>;

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
            
            // Exact cost check (can reference execution_context)
            if (filter.exact_cost.has_value() || filter.cost_ref.has_value()) {
                int required_cost = 0;
                if (filter.exact_cost.has_value()) {
                    required_cost = filter.exact_cost.value();
                } else if (filter.cost_ref.has_value() && execution_context) {
                    std::string key = filter.cost_ref.value();
                    if (execution_context->count(key)) {
                        required_cost = execution_context->at(key);
                    } else {
                        return false; // Reference not found, fail match
                    }
                }
                if (Props::get_cost(card_def) != required_cost) return false;
            }
            
            if (filter.min_power.has_value() && Props::get_power(card_def) < filter.min_power.value()) return false;

            if (filter.max_power.has_value()) {
                 if (Props::get_power(card_def) > filter.max_power.value()) return false;
            }
            // Step 5.2.3: Context Reference for Max Power
            if (filter.power_max_ref.has_value() && execution_context) {
                const auto& key = filter.power_max_ref.value();
                if (execution_context->count(key)) {
                    int max_val = execution_context->at(key);
                    if (Props::get_power(card_def) > max_val) return false;
                }
            }

            // 7. State Checks
            if (filter.is_tapped.has_value() && card.is_tapped != filter.is_tapped.value()) return false;
            if (filter.is_evolution.has_value()) {
                bool is_evo = Props::is_evolution(card_def);
                if (is_evo != filter.is_evolution.value()) return false;
            }
            if (filter.is_blocker.has_value()) {
                // Use state-aware check if possible
                bool is_blocker = false;
                if (!ignore_passives) {
                    is_blocker = has_keyword_simple(game_state, card, card_def, "BLOCKER");
                } else {
                    is_blocker = Props::has_keyword_intrinsic(card_def, "BLOCKER");
                }
                if (is_blocker != filter.is_blocker.value()) return false;
            }

            // 8. Composite AND Conditions (Step 3-2)
            if (!filter.and_conditions.empty()) {
                for (const auto& sub_filter : filter.and_conditions) {
                    if (!is_valid_target(card, card_def, sub_filter, game_state, source_controller, card_controller, ignore_passives, execution_context)) {
                        return false;
                    }
                }
            }

            return true;
        }

        // Helper to check Just Diver status specifically
        static bool is_protected_by_just_diver(const dm::core::CardInstance& card,
                                               const dm::core::CardDefinition& def,
                                               const dm::core::GameState& game_state,
                                               dm::core::PlayerID opponent_id) { // opponent_id is who is trying to target
             (void)opponent_id;
             if (!def.keywords.just_diver) return false;
             return game_state.turn_number == card.turn_played;
        }

        // NEW: Check for keywords including Passive Effects
        template<typename CardType>
        static bool has_keyword_simple(const dm::core::GameState& state,
                                       const dm::core::CardInstance& instance,
                                       const CardType& def,
                                       const std::string& keyword) {
             using Props = CardProperties<CardType>;
             // 1. Intrinsic
             if (Props::has_keyword_intrinsic(def, keyword)) return true;

             // 2. Passives
             // We implement a simplified filter check here to avoid recursion
             for (const auto& passive : state.passive_effects) {
                 bool match_type = false;
                 if (passive.type == dm::core::PassiveType::KEYWORD_GRANT && passive.str_value == keyword) match_type = true;
                 else if (passive.type == dm::core::PassiveType::BLOCKER_GRANT && keyword == "BLOCKER") match_type = true;
                 else if (passive.type == dm::core::PassiveType::SPEED_ATTACKER_GRANT && keyword == "SPEED_ATTACKER") match_type = true;
                 else if (passive.type == dm::core::PassiveType::SLAYER_GRANT && keyword == "SLAYER") match_type = true;

                 if (match_type) {
                     // Check target filter WITHOUT checking keywords recursively
                     // We use is_valid_target but we rely on the fact that GRANT_KEYWORD usually
                     // doesn't filter on the keyword it grants (circular).
                     // But if it filters on *other* keywords, we recurse.
                     // To be safe, we might want to pass a flag "ignore_passives" to is_valid_target?
                     // For now, assume standard usage.

                     // Need controllers
                     dm::core::PlayerID card_owner = 0;
                     if (instance.instance_id < (int)state.card_owner_map.size())
                         card_owner = state.get_card_owner(instance.instance_id);

                     // Avoid recursion: ignore_passives = true
                     if (is_valid_target(instance, def, passive.target_filter, state, passive.controller, card_owner, true, nullptr)) {
                         return true;
                     }
                 }
             }
             return false;
        }

        // NEW: Unified Can Attack Checks
        // Split into separate checks for precision

        static bool can_attack_creature(const dm::core::CardInstance& card,
                               const dm::core::CardDefinition& def,
                               const dm::core::GameState& game_state,
                               const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            (void)card_db;
            if (card.is_tapped) return false;

            if (card.summoning_sickness) {
                bool has_sa = has_keyword_simple(game_state, card, def, "SPEED_ATTACKER");
                bool is_evo = (def.type == dm::core::CardType::EVOLUTION_CREATURE) || def.keywords.evolution;
                bool has_mach = has_keyword_simple(game_state, card, def, "MACH_FIGHTER");

                // Can attack creatures if SA, Evo, or Mach Fighter
                if (!has_sa && !is_evo && !has_mach) {
                    return false;
                }
            }
            return true;
        }

        static bool can_attack_player(const dm::core::CardInstance& card,
                               const dm::core::CardDefinition& def,
                               const dm::core::GameState& game_state,
                               const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            (void)card_db;
            if (card.is_tapped) return false;

            if (card.summoning_sickness) {
                bool has_sa = has_keyword_simple(game_state, card, def, "SPEED_ATTACKER");
                bool is_evo = (def.type == dm::core::CardType::EVOLUTION_CREATURE) || def.keywords.evolution;

                // Mach Fighter does NOT allow attacking players
                if (!has_sa && !is_evo) {
                    return false;
                }
            }
            return true;
        }

        // Check if `blocker` can block `attacker`
        template<typename CardDefType>
        static bool can_be_blocked_by(const dm::core::CardInstance& attacker,
                                      const CardDefType& attacker_def,
                                      const dm::core::CardInstance& blocker,
                                      const CardDefType& blocker_def,
                                      const dm::core::GameState& game_state) {

            // 1. Basic Unblockable check
            if (has_keyword_simple(game_state, attacker, attacker_def, "UNBLOCKABLE")) {
                return false;
            }

            // 2. Passive Restrictions
            for (const auto& passive : game_state.passive_effects) {
                if (passive.type == dm::core::PassiveType::CANNOT_BLOCK) {
                    dm::core::PlayerID blocker_owner = 0;
                     if (blocker.instance_id < (int)game_state.card_owner_map.size())
                         blocker_owner = game_state.get_card_owner(blocker.instance_id);

                    if (is_valid_target(blocker, blocker_def, passive.target_filter, game_state, passive.controller, blocker_owner, true, nullptr)) {
                        return false;
                    }
                }

                // Case B: Passive on Attacker says "Cannot be blocked by Power X" (Not implemented in standard passives yet?)
                // Usually implemented as "Target creature cannot block this creature".
                // If the passive type is "CANNOT_BLOCK" but specifically targeted at "blockers of this creature"... complex.
            }

            return true;
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
        static bool match_civilization(const dm::core::CardDefinition& c, dm::core::Civilization target_civ) {
            return c.has_civilization(target_civ);
        }
        static bool match_type(const dm::core::CardDefinition& c, const std::string& type_str) {
            using namespace dm::core;
            if (type_str == "CARD") return true;
            if (type_str == "ELEMENT") {
                return c.type == CardType::CREATURE ||
                       c.type == CardType::EVOLUTION_CREATURE ||
                       c.type == CardType::CROSS_GEAR ||
                       c.type == CardType::TAMASEED ||
                       c.type == CardType::PSYCHIC_CREATURE ||
                       c.type == CardType::GR_CREATURE;
            }
            if (type_str == "TAMASEED") return c.type == CardType::TAMASEED;
            if (type_str == "CREATURE") return c.type == CardType::CREATURE || c.type == CardType::EVOLUTION_CREATURE;
            if (type_str == "SPELL") return c.type == CardType::SPELL;
            if (type_str == "EVOLUTION_CREATURE") return c.type == CardType::EVOLUTION_CREATURE;
            if (type_str == "CROSS_GEAR") return c.type == CardType::CROSS_GEAR;
            if (type_str == "CASTLE") return c.type == CardType::CASTLE;
            if (type_str == "PSYCHIC_CREATURE") return c.type == CardType::PSYCHIC_CREATURE;
            if (type_str == "GR_CREATURE") return c.type == CardType::GR_CREATURE;
            return false;
        }
        static bool is_evolution(const dm::core::CardDefinition& c) {
            return c.type == dm::core::CardType::EVOLUTION_CREATURE || c.keywords.evolution;
        }
        static bool is_blocker(const dm::core::CardDefinition& c) {
            return c.keywords.blocker;
        }
        static bool has_keyword_intrinsic(const dm::core::CardDefinition& c, const std::string& k) {
            if (k == "BLOCKER") return c.keywords.blocker;
            if (k == "SPEED_ATTACKER") return c.keywords.speed_attacker;
            if (k == "SLAYER") return c.keywords.slayer;
            if (k == "POWER_ATTACKER") return c.keywords.power_attacker;
            if (k == "MACH_FIGHTER") return c.keywords.mach_fighter;
            if (k == "SHIELD_TRIGGER") return c.keywords.shield_trigger;
            if (k == "UNBLOCKABLE") return c.keywords.unblockable;
            return false;
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
        static bool match_civilization(const dm::core::CardData& c, dm::core::Civilization target_civ) {
             return std::find(c.civilizations.begin(), c.civilizations.end(), target_civ) != c.civilizations.end();
        }
        static bool match_type(const dm::core::CardData& c, const std::string& type_str) {
            using namespace dm::core;
            if (type_str == "CARD") return true;
            if (type_str == "ELEMENT") {
                return c.type == CardType::CREATURE ||
                       c.type == CardType::EVOLUTION_CREATURE ||
                       c.type == CardType::CROSS_GEAR ||
                       c.type == CardType::TAMASEED ||
                       c.type == CardType::PSYCHIC_CREATURE ||
                       c.type == CardType::GR_CREATURE;
            }
            if (type_str == "TAMASEED") return c.type == CardType::TAMASEED;
            if (type_str == "CREATURE") return c.type == CardType::CREATURE || c.type == CardType::EVOLUTION_CREATURE;
            if (type_str == "SPELL") return c.type == CardType::SPELL;
            if (type_str == "EVOLUTION_CREATURE") return c.type == CardType::EVOLUTION_CREATURE;
            if (type_str == "CROSS_GEAR") return c.type == CardType::CROSS_GEAR;
            if (type_str == "CASTLE") return c.type == CardType::CASTLE;
            if (type_str == "PSYCHIC_CREATURE") return c.type == CardType::PSYCHIC_CREATURE;
            if (type_str == "GR_CREATURE") return c.type == CardType::GR_CREATURE;
            return false;
        }
        static bool is_evolution(const dm::core::CardData& c) {
            return c.type == dm::core::CardType::EVOLUTION_CREATURE;
        }
        static bool is_blocker(const dm::core::CardData& /*c*/) {
            return false;
        }
        static bool has_keyword_intrinsic(const dm::core::CardData& c, const std::string& k) {
             if (c.keywords.has_value() && c.keywords->count(k)) return true;
             return false;
        }
    };

}
