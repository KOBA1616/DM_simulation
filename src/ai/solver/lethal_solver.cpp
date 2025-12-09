#include "lethal_solver.hpp"
#include <algorithm>
#include <iostream>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;

    bool LethalSolver::is_lethal(const GameState& game_state,
                                 const std::map<CardID, CardDefinition>& card_db) {

        const Player& active_player = game_state.get_active_player();
        const Player& opponent = game_state.get_non_active_player();

        // 1. Gather Attackers
        std::vector<AttackerInfo> attackers;

        // On Board Attackers
        for (const auto& card : active_player.battle_zone) {
            // Check if can attack
            bool can_attack = !card.is_tapped;

            // Summoning Sickness check
            if (card.summoning_sickness) {
                // Check Speed Attacker
                bool has_sa = false;
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    if (def.keywords.speed_attacker || def.type == CardType::EVOLUTION_CREATURE) {
                        has_sa = true;
                    }
                }
                if (!has_sa) can_attack = false;
            }

            if (can_attack) {
                int power = 0;
                if (card_db.count(card.card_id)) {
                    power = card_db.at(card.card_id).power;
                }
                attackers.push_back({card.instance_id, power, true});
            }
        }

        // 2. Gather Blockers
        std::vector<BlockerInfo> blockers;
        for (const auto& card : opponent.battle_zone) {
            // Must be untapped to block (usually)
            if (!card.is_tapped) {
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    if (def.keywords.blocker) {
                        blockers.push_back({card.instance_id, def.power});
                    }
                }
            }
        }

        // 3. Simple Lethal Check
        int shields_count = opponent.shield_zone.size();
        int required_attacks = shields_count + blockers.size() + 1;

        if (attackers.size() >= (size_t)required_attacks) {
            return true;
        }

        return false;
    }

}
