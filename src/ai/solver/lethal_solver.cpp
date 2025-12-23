#include "lethal_solver.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/passive_effect_system.hpp"

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;

    // Helper struct for detailed attacker info
    struct AttackerDetail {
        int instance_id;
        int power;
        int breaker_count; // 1 = Single, 2 = Double, 3 = Triple
        bool is_unblockable;
        bool can_attack_creatures; // For future use (clearing blockers)
    };

    struct BlockerDetail {
        int instance_id;
        int power;
    };

    bool LethalSolver::is_lethal(const GameState& game_state,
                                 const std::map<CardID, CardDefinition>& card_db) {

        const Player& active_player = game_state.players[game_state.active_player_id];
        const Player& opponent = game_state.players[1 - game_state.active_player_id];

        // 1. Gather Attackers
        std::vector<AttackerDetail> attackers;

        for (const auto& card : active_player.battle_zone) {
            // Definition Check
            if (!card_db.count(card.card_id)) continue;
            const auto& def = card_db.at(card.card_id);

            // Use Unified Logic - Strict Player Attack Check
            bool can_attack = TargetUtils::can_attack_player(card, def, game_state, card_db);

            // Manual Passive Check (since TargetUtils cannot depend on PassiveEffectSystem)
            if (can_attack) {
                if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_ATTACK, card_db)) {
                    can_attack = false;
                }
            }

            if (!can_attack) continue;

            AttackerDetail info;
            info.instance_id = card.instance_id;
            info.power = def.power;

            // Breaker Capability
            info.breaker_count = 1;
            if (def.keywords.has(Keyword::WORLD_BREAKER)) info.breaker_count = 999;
            else if (def.keywords.has(Keyword::TRIPLE_BREAKER)) info.breaker_count = 3;
            else if (def.keywords.has(Keyword::DOUBLE_BREAKER)) info.breaker_count = 2;

            // Unblockable Capability
            info.is_unblockable = def.keywords.has(Keyword::UNBLOCKABLE);

            attackers.push_back(info);
        }

        // 2. Gather Blockers
        std::vector<BlockerDetail> blockers;
        for (const auto& card : opponent.battle_zone) {
            if (!card.is_tapped) {
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    if (def.keywords.has(Keyword::BLOCKER)) {
                        blockers.push_back({card.instance_id, def.power});
                    }
                }
            }
        }

        int blockers_count = blockers.size();
        int opponent_shields = opponent.shield_zone.size();

        // 3. Simulation Logic
        // Strategy:
        // Attackers want to maximize shield damage to reach direct attack.
        // Defenders want to minimize shield damage or prevent direct attack.
        // Assuming "Greedy" play from both sides for Solver.

        // Sort Attackers:
        // Priority 1: Unblockable (Guaranteed damage) - put them effectively last?
        // No, put High Breakers first to force blocks.
        // If we have Unblockable Breakers, they guarantee breaks.
        // If we have Blockable Breakers, they consume blockers.

        // Let's sort blockable attackers by Breaker Count (Desc).
        // Let's keep unblockable separately or mark them.

        std::vector<AttackerDetail> blockable_attackers;
        std::vector<AttackerDetail> unblockable_attackers;

        for (const auto& att : attackers) {
            if (att.is_unblockable) {
                unblockable_attackers.push_back(att);
            } else {
                blockable_attackers.push_back(att);
            }
        }

        // Sort blockable attackers: Highest breakers first.
        std::sort(blockable_attackers.begin(), blockable_attackers.end(), [](const AttackerDetail& a, const AttackerDetail& b) {
            return a.breaker_count > b.breaker_count;
        });

        // Resolve Blocks
        // Defender strategy: Block the highest threat.
        // "Threat" usually means Breaker Count.
        // So Defender consumes blockers against the first N blockable attackers.

        std::vector<AttackerDetail> successful_attackers;

        // Add all unblockables (they are always successful)
        successful_attackers.insert(successful_attackers.end(), unblockable_attackers.begin(), unblockable_attackers.end());

        // Process blockables
        for (const auto& att : blockable_attackers) {
            if (blockers_count > 0) {
                // Blocked!
                // Assumption: Blocker defeats Attacker or stops it.
                // We ignore "Blocker destruction" or "Slayer" for now in this Greedy Solver.
                // Just assume the attack is stopped.
                blockers_count--;
            } else {
                // Not blocked
                successful_attackers.push_back(att);
            }
        }

        // 4. Resolve Breaks
        // Now we have a list of attackers that hit.
        // We need to see if they can clear shields and land a direct attack.

        std::sort(successful_attackers.begin(), successful_attackers.end(), [](const AttackerDetail& a, const AttackerDetail& b) {
            return a.breaker_count > b.breaker_count;
        });

        int current_shields = opponent_shields;
        int direct_attacks_landed = 0;

        for (const auto& att : successful_attackers) {
            if (current_shields > 0) {
                int breaks = std::min(current_shields, att.breaker_count);
                current_shields -= breaks;

                // Note: If breaker_count > current_shields, the excess is lost (doesn't carry over to players).
                // But wait, if shields become 0 *during* this attack, this attacker does NOT continue to player.
                // The attack resolves on shields.
            } else {
                // Shields are 0. Direct Attack!
                direct_attacks_landed++;
            }
        }

        return direct_attacks_landed > 0;
    }

}
