#include "lethal_solver.hpp"
#include <algorithm>
#include <iostream>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;

    // Structs are defined in lethal_solver.hpp (private members of LethalSolver)

    bool LethalSolver::is_lethal(const GameState& game_state,
                                 const std::map<CardID, CardDefinition>& card_db) {

        const Player& active_player = game_state.get_active_player();
        const Player& opponent = game_state.get_non_active_player();

        // 1. Gather Attackers
        std::vector<AttackerInfo> attackers;

        for (const auto& card : active_player.battle_zone) {
            bool can_attack = !card.is_tapped;

            int power = 0;
            int breaks = 1;
            bool is_unblockable = false;
            // Additional checks
            if (card_db.count(card.card_id)) {
                const auto& def = card_db.at(card.card_id);
                power = def.power;
                if (def.keywords.double_breaker) breaks = 2;
                if (def.keywords.triple_breaker) breaks = 3;
                // if (def.keywords.world_breaker) breaks = 99; // TODO: Add World Breaker

                if (card.summoning_sickness) {
                    bool has_sa = def.keywords.speed_attacker || (def.type == CardType::EVOLUTION_CREATURE);
                    if (!has_sa) can_attack = false;
                }
            } else {
                 if (card.summoning_sickness) can_attack = false;
            }

            if (can_attack) {
                attackers.push_back({card.instance_id, power, breaks, is_unblockable, true});
            }
        }

        // 2. Gather Blockers
        std::vector<BlockerInfo> blockers;
        for (const auto& card : opponent.battle_zone) {
            if (!card.is_tapped) {
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    if (def.keywords.blocker) {
                        blockers.push_back({card.instance_id, def.power});
                    }
                }
            }
        }

        // 3. Simulation Logic (Phase 1.1)

        // Sort attackers by 'breaks' descending to assume optimal usage (or optimal blocking target).
        // Strategy:
        // We want to verify if there exists a subset of unblocked attackers that can clear shields AND have 1 left.
        // Opponent blocks to minimize our damage.
        // Opponent Strategy (Greedy): Block the attackers with highest 'breaks'.

        // Sort attackers: Highest breaks first.
        std::sort(attackers.begin(), attackers.end(), [](const AttackerInfo& a, const AttackerInfo& b) {
            return a.breaks > b.breaks;
        });

        // Simulate Blocking
        // Blockers remove the top N attackers (assuming they can block them).
        // Note: Ignoring power comparisons for simplicity in Phase 1.1 (Assume blockers win or stop attack).
        // Note: Ignoring 'unblockable' for now as keyword isn't fully exposed, but structure supports it.

        size_t blocked_count = blockers.size();
        std::vector<AttackerInfo> unblocked_attackers;

        for (const auto& att : attackers) {
            if (blocked_count > 0) {
                // This attacker is blocked.
                blocked_count--;
            } else {
                unblocked_attackers.push_back(att);
            }
        }

        if (unblocked_attackers.empty()) {
            return false;
        }

        // Now we have unblocked attackers.
        // We need to break 'shields_count' shields.
        // And have at least 1 attacker remaining for the Direct Attack.

        int shields_count = opponent.shield_zone.size();

        // Strategy for us: Use highest breakers to clear shields efficiently.
        // The list is already sorted by breaks DESC.

        int damage_dealt = 0;
        size_t used_attackers = 0;

        for (const auto& att : unblocked_attackers) {
            // If we have already broken all shields, we just need to confirm we have this attacker left.
            if (damage_dealt >= shields_count) {
                // Shields are gone. We have an attacker 'att' ready to direct attack.
                return true;
            }

            // Use this attacker to break shields
            damage_dealt += att.breaks;
            used_attackers++;
        }

        // If loop finishes and damage_dealt >= shields_count,
        // it means we used ALL attackers to break shields (or overkill).
        // But we need one MORE for direct attack?
        // Wait. If damage_dealt >= shields_count was achieved at the very last attacker,
        // then we have 0 attackers left for direct attack.
        // The check inside the loop returns true ONLY if we enter the loop AND shields are ALREADY gone.
        // So if we break the last shield with the last attacker, the next iteration would trigger 'true'.
        // But there is no next iteration.

        // So if loop finishes, we failed to have a surplus attacker.

        return false;
    }

}
