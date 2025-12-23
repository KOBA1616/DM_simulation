#include "lethal_solver.hpp"
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/passive_effect_system.hpp"

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;

    // Search Structures
    struct SimAttacker {
        int index; // Bit index in mask
        int power;
        int breaker; // 999 for World Breaker
        bool unblockable;
        // Future: bool slayer, bool untaps_after_win, etc.
    };

    struct SimBlocker {
        int index; // Bit index in mask
        int power;
        // Future: bool slayer, bool blocks_multiple, etc.
    };

    // State key for memoization
    struct SearchState {
        uint64_t attacker_mask;
        uint64_t blocker_mask;
        int shields;

        bool operator<(const SearchState& other) const {
            if (attacker_mask != other.attacker_mask) return attacker_mask < other.attacker_mask;
            if (blocker_mask != other.blocker_mask) return blocker_mask < other.blocker_mask;
            return shields < other.shields;
        }
    };

    class LethalDFS {
        const std::vector<SimAttacker>& attackers;
        const std::vector<SimBlocker>& blockers;
        std::map<SearchState, bool> memo;
        int initial_shields;

    public:
        LethalDFS(const std::vector<SimAttacker>& a, const std::vector<SimBlocker>& b, int shields)
            : attackers(a), blockers(b), initial_shields(shields) {}

        bool solve(SearchState state) {
            // Check memoization
            auto it = memo.find(state);
            if (it != memo.end()) return it->second;

            // Base case: No attackers left
            if (state.attacker_mask == 0) {
                // If we haven't won yet (shields <= 0 checked at transition), we lose this branch
                return false;
            }

            // Max Node (Attacker Choice)
            // We want to find AT LEAST ONE attack that guarantees a win.

            bool can_force_win = false;

            for (const auto& att : attackers) {
                if (!((state.attacker_mask >> att.index) & 1)) continue;

                if (simulate_attack(att, state)) {
                    can_force_win = true;
                    break; // Pruning: Found a winning move
                }
            }

            memo[state] = can_force_win;
            return can_force_win;
        }

        bool simulate_attack(const SimAttacker& att, SearchState state) {
            // Min Node (Opponent Choice)
            // Opponent wants to find AT LEAST ONE response that prevents our win.
            // If they find a survival path, this attack fails (return false).
            // If ALL responses lead to our win, return true.

            // 1. Path: No Block
            {
                // Check direct attack condition
                if (state.shields <= 0) {
                    // Direct attack successful -> WIN
                    // This path leads to Win.
                    // Opponent cannot choose this to survive.
                    // So we treat this as "Opponent cannot survive via No Block".
                    // (Proceed to check Block)
                } else {
                    // Shield Break
                    SearchState next_state = state;
                    next_state.attacker_mask &= ~(1ULL << att.index); // Attacker taps/used

                    int damage = att.breaker;
                    if (damage >= 999) next_state.shields = 0; // World Breaker
                    else next_state.shields = std::max(0, state.shields - damage);

                    // Recursively check if we can win from the post-break state
                    if (!solve(next_state)) {
                        return false; // Opponent survives by taking the hit
                    }
                }
            }

            // If unblockable, No Block is the only option.
            // Since we passed the check above (didn't return false), it means No Block leads to Win.
            if (att.unblockable) return true;

            // 2. Path: Block with eligible blocker
            for (const auto& blk : blockers) {
                if (!((state.blocker_mask >> blk.index) & 1)) continue;

                // Assumption: Filter logic already verified this blocker can block.
                // In future, check "can block this specific creature" here.

                SearchState next_state = state;
                next_state.attacker_mask &= ~(1ULL << att.index); // Attacker taps
                next_state.blocker_mask &= ~(1ULL << blk.index);  // Blocker taps

                // Combat Logic (Who dies?)
                // Currently, death implies removal from mask for NEXT turns.
                // But taps also imply removal from mask.
                // So effectively both are removed from the available pool for this turn.
                // (Unless "Untap" abilities exist).
                // For this implementation, removing from mask is correct.

                if (!solve(next_state)) {
                    return false; // Opponent survives by blocking with 'blk'
                }
            }

            // If we get here, it means Opponent cannot survive via No Block AND cannot survive via any Block.
            return true;
        }
    };

    bool LethalSolver::is_lethal(const GameState& game_state,
                                 const std::map<CardID, CardDefinition>& card_db) {

        const Player& active_player = game_state.players[game_state.active_player_id];
        const Player& opponent = game_state.players[1 - game_state.active_player_id];

        // 1. Gather Attackers
        std::vector<SimAttacker> attackers;
        int att_index = 0;

        for (const auto& card : active_player.battle_zone) {
            if (!card_db.count(card.card_id)) continue;
            const auto& def = card_db.at(card.card_id);

            bool can_attack = TargetUtils::can_attack_player(card, def, game_state, card_db);

            // Manual Passive Check
            if (can_attack) {
                if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_ATTACK, card_db)) {
                    can_attack = false;
                }
            }

            if (!can_attack) continue;

            SimAttacker info;
            info.index = att_index++;
            info.power = def.power;

            info.breaker = 1;
            if (def.keywords.world_breaker) info.breaker = 999;
            else if (def.keywords.triple_breaker) info.breaker = 3;
            else if (def.keywords.double_breaker) info.breaker = 2;

            info.unblockable = def.keywords.unblockable;

            attackers.push_back(info);
            if (att_index >= 64) break; // Safety limit
        }

        // 2. Gather Blockers
        std::vector<SimBlocker> blockers;
        int blk_index = 0;

        for (const auto& card : opponent.battle_zone) {
            if (!card.is_tapped) {
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    if (def.keywords.blocker) {
                        SimBlocker info;
                        info.index = blk_index++;
                        info.power = def.power;
                        blockers.push_back(info);
                        if (blk_index >= 64) break;
                    }
                }
            }
        }

        int opponent_shields = opponent.shield_zone.size();

        // 3. Run Search
        SearchState initial_state;
        initial_state.attacker_mask = (attackers.size() == 64) ? ~0ULL : ((1ULL << attackers.size()) - 1);
        initial_state.blocker_mask = (blockers.size() == 64) ? ~0ULL : ((1ULL << blockers.size()) - 1);
        initial_state.shields = opponent_shields;

        LethalDFS dfs(attackers, blockers, opponent_shields);
        return dfs.solve(initial_state);
    }

}
