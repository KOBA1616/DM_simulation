#include "restriction_system.hpp"
#include "engine/systems/effects/passive_effect_system.hpp"
#include "engine/utils/target_utils.hpp"
#include "core/game_state.hpp"

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
             // Note: Some effects might allow attacking untapped creatures, but we enforce standard rules for now
             if (target_card && !target_card->is_tapped) {
                 // TODO: Check for "can attack untapped creatures" effects
                 // For now, in original code it returns (aborts)
                 return true;
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
