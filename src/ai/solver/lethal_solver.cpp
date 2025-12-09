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

        // Debug prints
        // std::cout << "Lethal Check: Active P" << (int)active_player.id << " vs P" << (int)opponent.id << std::endl;
        // std::cout << "Active BZone Size: " << active_player.battle_zone.size() << std::endl;
        // std::cout << "Opp Shields: " << opponent.shield_zone.size() << std::endl;

        // 1. Gather Attackers
        std::vector<AttackerInfo> attackers;

        // On Board Attackers
        for (const auto& card : active_player.battle_zone) {
            // Check if can attack
            bool can_attack = !card.is_tapped;

            // Summoning Sickness check
            if (card.summoning_sickness) {
                // std::cout << "Card " << card.card_id << " is sick." << std::endl;
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

            // Check attack restrictions (Can't attack players) - TODO
            // For now assume all creatures can attack players unless specified otherwise.

            if (can_attack) {
                int power = 0; // Needed for power comparisons with blockers
                if (card_db.count(card.card_id)) {
                    power = card_db.at(card.card_id).power; // Use base power for now
                }
                attackers.push_back({card.instance_id, power, true});
            } else {
                // std::cout << "Card " << card.card_id << " cannot attack." << std::endl;
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
        // Can we break all shields + direct attack?
        // Attackers needed = Opponent Shields + Opponent Blockers + 1 (Direct Attack)

        // This is a naive heuristic (Count Only).
        // It assumes every blocker can stop exactly one attack.
        // It ignores Blockers losing battles (Slayer, High Power) for now.
        // It ignores Double Breakers.

        // Improvement: Consider breakers
        // For Step 1 (Basic): Just count.

        int shields_count = opponent.shield_zone.size();
        int required_attacks = shields_count + blockers.size() + 1;

        // std::cout << "Attackers: " << attackers.size() << " Required: " << required_attacks << std::endl;

        if (attackers.size() >= (size_t)required_attacks) {
            return true;
        }

        // TODO: Detailed simulation (Breaker logic, Power matching) in Phase 1.1

        return false;
    }

}
