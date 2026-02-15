#include "lethal_solver.hpp"
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include "engine/systems/game_logic_system.hpp"
#include "engine/actions/intent_generator.hpp"

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;
    using namespace dm::engine::systems;

    static bool is_useful_creature(const CardDefinition& def) {
        if (def.keywords.speed_attacker) return true;
        if (def.keywords.evolution) return true;
        if (def.keywords.g_zero) return true;
        if (def.keywords.revolution_change) return true;
        if (def.keywords.cip) return true;

        // Scan effects for ON_PLAY
        for (const auto& eff : def.effects) {
            if (eff.trigger == TriggerType::ON_PLAY) return true;
        }
        return false;
    }

    // Helper to check if an action is relevant for lethal
    static bool is_aggressive_action(const CommandDef& action, const GameState& state, const std::map<CardID, CardDefinition>& card_db) {
        switch (action.type) {
            case CommandType::ATTACK_PLAYER:
                return true;
            case CommandType::ATTACK_CREATURE:
                // Removing blockers is good
                return true;
            case CommandType::BREAK_SHIELD:
                return true;

            case CommandType::PLAY_FROM_ZONE:
                if (card_db.count(action.instance_id)) { // Note: card_id not directly in CommandDef? instance_id can lookup.
                     const auto* inst = state.get_card_instance(action.instance_id);
                     if (inst && card_db.count(inst->card_id)) {
                         const auto& def = card_db.at(inst->card_id);
                         if (def.type == CardType::CREATURE) {
                             if (is_useful_creature(def)) return true;
                         }
                         if (def.type == CardType::SPELL) return true;
                     }
                }
                return false;

            case CommandType::USE_ABILITY:
                return true;

            // Intermediate / Atomic
            case CommandType::RESOLVE_BATTLE:
            case CommandType::RESOLVE_PLAY:
            case CommandType::SHIELD_TRIGGER:
            case CommandType::SELECT_TARGET:
            case CommandType::RESOLVE_EFFECT:
            case CommandType::SELECT_NUMBER:
            case CommandType::CHOICE:
                return true;

            case CommandType::BLOCK:
                return true;

            case CommandType::PASS:
                if (state.current_phase == Phase::MAIN) return true;
                if (state.current_phase == Phase::BLOCK) return true; // Opponent passing block -> Good

                return false;

            default:
                return false;
        }
    }

    class LethalSearch {
        const std::map<CardID, CardDefinition>& card_db;
        PlayerID root_player;
        int max_depth;

    public:
        LethalSearch(const std::map<CardID, CardDefinition>& db, PlayerID player, int depth)
            : card_db(db), root_player(player), max_depth(depth) {}

        bool search(GameState state, int depth) {
            if (state.winner != GameResult::NONE) {
                return state.winner == static_cast<GameResult>(root_player + 1);
            }

            if (depth >= max_depth) return false;

            std::vector<CommandDef> actions = IntentGenerator::generate_legal_commands(state, card_db);

            if (actions.empty()) {
                return false;
            }

            // Determine who is making the decision
            PlayerID acting_player = state.active_player_id;
            if (state.current_phase == Phase::BLOCK) {
                acting_player = 1 - state.active_player_id;
            }

            bool is_root_decision = (acting_player == root_player);
            bool opponent_can_survive = false;

            for (const auto& action : actions) {
                if (is_root_decision) {
                    if (action.type == CommandType::MANA_CHARGE) continue;

                    // Allow PLAY explicitly with check
                    if (action.type == CommandType::PLAY_FROM_ZONE) {
                         const auto* inst = state.get_card_instance(action.instance_id);
                         if (inst && card_db.count(inst->card_id)) {
                             const auto& def = card_db.at(inst->card_id);
                             bool aggro = false;
                             if (def.type == CardType::CREATURE) {
                                 if (is_useful_creature(def)) aggro = true;
                             } else if (def.type == CardType::SPELL) {
                                 aggro = true;
                             }
                             if (!aggro) {
                                 continue;
                             }
                         }
                    } else if (!is_aggressive_action(action, state, card_db)) {
                        continue;
                    }
                }

                GameState next_state = state.clone();
                GameLogicSystem::resolve_command_oneshot(next_state, action, card_db);

                bool result = search(std::move(next_state), depth + 1);

                if (is_root_decision) {
                    if (result) return true; // Max Node: Found a win
                } else {
                    if (!result) {
                        opponent_can_survive = true;
                        // Optimization: If it's opponent's turn and they found a way to survive,
                        // then this branch (the move leading to this state) is failed for ROOT.
                        // We need ALL opponent moves to lead to ROOT win.
                        return false;
                    }
                }
            }

            if (is_root_decision) return false; // Tried all moves, none won.

            // If Opponent Decision (Min Node):
            // We only reach here if loop finished without returning false.
            // Means ALL opponent moves led to result=true (Win).
            return true;
        }
    };

    bool LethalSolver::is_lethal(const GameState& initial_state, const std::map<CardID, CardDefinition>& card_db) {
        if (initial_state.winner != GameResult::NONE) return false;
        LethalSearch solver(card_db, initial_state.active_player_id, 20);
        return solver.search(initial_state.clone(), 0);
    }

}
