#include "restriction_system.hpp"
#include "engine/systems/effects/passive_effect_system.hpp"
#include "engine/utils/target_utils.hpp"
#include "core/game_state.hpp"
#include <filesystem>
#include <fstream>

namespace dm::engine::systems {

    using namespace core;

    bool RestrictionSystem::is_play_forbidden(const GameState& state,
                                              const CardInstance& card,
                                              const CardDefinition& def,
                                              const std::string& origin_zone,
                                              const std::map<CardID, CardDefinition>& card_db) {

        // 1. Spell Restrictions
        if (def.type == CardType::SPELL) {
             if (PassiveEffectSystem::instance().check_restriction(state, card, PassiveType::CANNOT_USE_SPELLS, card_db)) return true;
             if (PassiveEffectSystem::instance().check_restriction(state, card, PassiveType::LOCK_SPELL_BY_COST, card_db)) return true;
        }

        // 2. Summon Restrictions (CANNOT_SUMMON)
        for (const auto& eff : state.passive_effects) {
            if (eff.type == PassiveType::CANNOT_SUMMON && (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE)) {
                 // Check Origin match
                 bool origin_match = true;
                 if (!eff.target_filter.zones.empty()) {
                     if (origin_zone.empty()) {
                         // If unknown origin, we assume safe or skip?
                         // In original code: if origin_str.empty() { match = false }
                         origin_match = false;
                     } else {
                         origin_match = false;
                         for (const auto& z : eff.target_filter.zones) {
                             if (z == origin_zone) { origin_match = true; break; }
                         }
                     }
                 }
                 // If zones empty, applies to all -> origin_match true

                 if (!origin_match) continue;

                 // Check Card match
                 FilterDef check_filter = eff.target_filter;
                 check_filter.zones.clear(); // Already checked

                 if (dm::engine::utils::TargetUtils::is_valid_target(card, def, check_filter, state, eff.controller, card.owner, true)) {
                     // Prohibited
                     return true;
                 }
            }
        }

        // 3. Limit Put Creature Restrictions (LIMIT_PUT_CREATURE_PER_TURN)
        if (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE) {
            int strict_limit = -1;
            for (const auto& eff : state.passive_effects) {
                if (eff.type == PassiveType::LIMIT_PUT_CREATURE_PER_TURN) {
                    if (dm::engine::utils::TargetUtils::is_valid_target(card, def, eff.target_filter, state, eff.controller, card.owner, true)) {
                        int current_limit = eff.value;
                        if (strict_limit == -1 || current_limit < strict_limit) {
                            strict_limit = current_limit;
                        }
                    }
                }
            }
            if (strict_limit != -1) {
                if (state.turn_stats.creatures_played_this_turn >= strict_limit) {
                    return true; // Limit reached
                }
            }
        }

        return false;
    }

    bool RestrictionSystem::is_attack_forbidden(const GameState& state,
                                                const CardInstance& attacker,
                                                const CardDefinition& def,
                                                int target_id,
                                                const std::map<CardID, CardDefinition>& card_db) {

        bool is_player_attack = (target_id == -1);
         if (is_player_attack) {
             if (!dm::engine::utils::TargetUtils::can_attack_player(attacker, def, state, card_db)) return true;
         } else {
             if (!dm::engine::utils::TargetUtils::can_attack_creature(attacker, def, state, card_db)) return true;

             // Check if target is valid (Standard Rule: Must be tapped)
             const CardInstance* target_card = state.get_card_instance(target_id);
            if (target_card && !target_card->is_tapped) {
                // Standard rule: can't attack untapped creatures unless an effect allows it.
                // If there is a passive effect ALLOW_ATTACK_UNTAPPED applying to attacker,
                // allow the attack; otherwise forbid.
                bool allowed = PassiveEffectSystem::instance().allows_attack_untapped(state, attacker, card_db);
                try {
                    std::filesystem::create_directories("logs");
                    std::ofstream ofs("logs/restriction_debug.txt", std::ios::app);
                    if (ofs) {
                        ofs << "[Restriction] attacker=" << attacker.instance_id
                            << " target=" << target_id
                            << " target_tapped=" << (target_card->is_tapped?1:0)
                            << " allows_attack_untapped=" << (allowed?1:0)
                            << "\n";
                    }
                } catch(...) {}

                if (!allowed) {
                    return true;
                }
            }
         }

         if (PassiveEffectSystem::instance().check_restriction(state, attacker, PassiveType::CANNOT_ATTACK, card_db)) return true;

         return false;
    }

    bool RestrictionSystem::is_block_forbidden(const GameState& state,
                                               const CardInstance& blocker,
                                               const CardDefinition& def,
                                               const std::map<CardID, CardDefinition>& card_db) {

        if (!dm::engine::utils::TargetUtils::has_keyword_simple(state, blocker, def, "BLOCKER")) return true;
        if (blocker.is_tapped) return true;
        if (PassiveEffectSystem::instance().check_restriction(state, blocker, PassiveType::CANNOT_BLOCK, card_db)) return true;

        return false;
    }

}
