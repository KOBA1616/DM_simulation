#include "phase_strategies.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/mechanics/mana_system.hpp"
#include "engine/systems/effects/passive_effect_system.hpp"
#include "engine/systems/mechanics/cost_payment_system.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace dm::engine {

    using namespace dm::core;

    std::vector<CommandDef> ManaPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<CommandDef> actions;
        const auto& game_state = ctx.game_state;
        const Player& active_player = game_state.players[game_state.active_player_id];

        try {
            std::filesystem::create_directories("logs");
            std::ofstream ofs("logs/mana_phase_debug.txt", std::ios::app);
            if (ofs) {
                ofs << "[ManaPhaseStrategy] mana_charged_this_turn=" 
                    << (game_state.turn_stats.mana_charged_by_player[game_state.active_player_id] ? "TRUE" : "FALSE")
                    << " turn=" << game_state.turn_number
                    << " hand_size=" << active_player.hand.size()
                    << " active_pid=" << (int)game_state.active_player_id << "\n";
            }
        } catch(...) {}

        if (!game_state.turn_stats.mana_charged_by_player[game_state.active_player_id]) {
            for (size_t i = 0; i < active_player.hand.size(); ++i) {
                const auto& card = active_player.hand[i];
                CommandDef cmd;
                cmd.type = CommandType::MANA_CHARGE;
                cmd.instance_id = card.instance_id;
                cmd.slot_index = static_cast<int>(i);
                actions.push_back(cmd);
            }
            
            try {
                std::ofstream ofs("logs/mana_phase_debug.txt", std::ios::app);
                if (ofs) {
                    ofs << "[ManaPhaseStrategy] Generated " << actions.size() << " MANA_CHARGE actions\n";
                }
            } catch(...) {}
            
            CommandDef pass;
            pass.type = CommandType::PASS;
            actions.push_back(pass);
        } else {
            try {
                std::ofstream ofs("logs/mana_phase_debug.txt", std::ios::app);
                if (ofs) {
                    ofs << "[ManaPhaseStrategy] Already charged - returning PASS only\n";
                }
            } catch(...) {}
            
            CommandDef pass;
            pass.type = CommandType::PASS;
            actions.push_back(pass);
        }

        return actions;
    }

    std::vector<CommandDef> MainPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<CommandDef> actions;
        const auto& game_state = ctx.game_state;
        const auto& card_db = ctx.card_db;
        const Player& active_player = game_state.players[game_state.active_player_id];

        try {
            std::filesystem::create_directories("logs");
            std::ofstream enter_ofs("logs/main_phase_checks.txt", std::ios::app);
            if (enter_ofs) {
                enter_ofs << "[MainPhaseEnter] player=" << static_cast<int>(game_state.active_player_id)
                          << " turn=" << game_state.turn_number
                          << " hand_count=" << active_player.hand.size() << "\n";
            }
        } catch (...) {}

        auto log_skip = [&](int hand_index, const CardInstance& card, const CardDefinition& def, bool spell_restricted, int adjusted_cost, int available_mana, bool can_pay) {
            try {
                std::filesystem::create_directories("logs");
                std::ofstream ofs("logs/main_phase_checks.txt", std::ios::app);
                if (!ofs) return;
                std::ostringstream ss;
                ss << "[MainPhase] player=" << static_cast<int>(game_state.active_player_id)
                   << " turn=" << game_state.turn_number
                   << " hand_idx=" << hand_index
                   << " instance_id=" << card.instance_id
                   << " card_id=" << def.id
                   << " adjusted_cost=" << adjusted_cost
                   << " available_mana=" << available_mana
                   << " spell_restricted=" << (spell_restricted?1:0)
                   << " can_pay=" << (can_pay?1:0)
                   << "\n";
                ofs << ss.str();
            } catch (...) {}
        };

        for (size_t i = 0; i < active_player.hand.size(); ++i) {
            const auto& card = active_player.hand[i];
            if (card_db.count(card.card_id)) {
                const auto& def = card_db.at(card.card_id);

                bool spell_restricted = false;
                if (def.type == CardType::SPELL) {
                    if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_USE_SPELLS, card_db)) {
                        spell_restricted = true;
                    }
                    if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::LOCK_SPELL_BY_COST, card_db)) {
                        spell_restricted = true;
                    }
                }

                // 1. Standard Play
                {
                    int adjusted_cost = ManaSystem::get_adjusted_cost(game_state, active_player, def);
                    int available_mana = ManaSystem::get_usable_mana_count(game_state, active_player.id, def.civilizations, card_db);
                    bool can_pay = ManaSystem::can_pay_cost(game_state, active_player, def, card_db);

                    if (!spell_restricted && can_pay) {
                        CommandDef cmd;
                        cmd.type = CommandType::PLAY_FROM_ZONE;
                        cmd.instance_id = card.instance_id;
                        cmd.amount = 0;
                        cmd.slot_index = static_cast<int>(i);
                        actions.push_back(cmd);
                    } else {
                        log_skip(static_cast<int>(i), card, def, spell_restricted, adjusted_cost, available_mana, can_pay);
                    }
                }

                // 2. Twinpact Spell Side Play
                if (def.spell_side) {
                    const auto& spell_def = *def.spell_side;
                    bool side_restricted = false;
                    if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_USE_SPELLS, card_db)) {
                        side_restricted = true;
                    }
                    if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::LOCK_SPELL_BY_COST, card_db)) {
                        side_restricted = true;
                    }

                    {
                        int adjusted_cost_side = ManaSystem::get_adjusted_cost(game_state, active_player, spell_def);
                        int available_mana_side = ManaSystem::get_usable_mana_count(game_state, active_player.id, spell_def.civilizations, card_db);
                        bool can_pay_side = ManaSystem::can_pay_cost(game_state, active_player, spell_def, card_db);
                        if (!side_restricted && can_pay_side) {
                            CommandDef cmd;
                            cmd.type = CommandType::PLAY_FROM_ZONE;
                            cmd.instance_id = card.instance_id;
                            cmd.amount = 1;
                            cmd.slot_index = static_cast<int>(i);
                            actions.push_back(cmd);
                        } else {
                            log_skip(static_cast<int>(i), card, spell_def, side_restricted, adjusted_cost_side, available_mana_side, can_pay_side);
                        }
                    }
                }

                // 3. Active Cost Reductions
                for (const auto& reduction : def.cost_reductions) {
                    if (reduction.type == ReductionType::ACTIVE_PAYMENT) {
                        int max_units = CostPaymentSystem::calculate_max_units(game_state, active_player.id, reduction, card_db);

                        for (int units = 1; units <= max_units; ++units) {
                            if (reduction.max_units != -1 && units > reduction.max_units) break;

                            int reduction_val = units * reduction.reduction_amount;
                            int adjusted_cost = ManaSystem::get_adjusted_cost(game_state, active_player, def);
                            int effective_cost = std::max(reduction.min_mana_cost, adjusted_cost - reduction_val);

                            int available_mana = ManaSystem::get_usable_mana_count(game_state, active_player.id, def.civilizations, card_db);

                            if (available_mana >= effective_cost) {
                                CommandDef cmd;
                                cmd.type = CommandType::PLAY_FROM_ZONE;
                                cmd.instance_id = card.instance_id;
                                cmd.target_instance = units;
                                cmd.str_param = "ACTIVE_PAYMENT";
                                cmd.slot_index = static_cast<int>(i);
                                cmd.target_slot_index = units; // Mirror units to target_slot_index just in case
                                actions.push_back(cmd);
                            }
                        }
                    }
                }
            }
        }

        CommandDef pass_action;
        pass_action.type = CommandType::PASS;
        actions.push_back(pass_action);

        return actions;
    }

    std::vector<CommandDef> AttackPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<CommandDef> actions;
        const auto& game_state = ctx.game_state;
        const auto& card_db = ctx.card_db;
        const Player& active_player = game_state.players[game_state.active_player_id];
        const Player& opponent = game_state.players[1 - game_state.active_player_id];

        for (size_t i = 0; i < active_player.battle_zone.size(); ++i) {
            const auto& card = active_player.battle_zone[i];

            if (card_db.count(card.card_id)) {
                const auto& def = card_db.at(card.card_id);

                bool can_attack_player = TargetUtils::can_attack_player(card, def, game_state, card_db);
                bool can_attack_creature = TargetUtils::can_attack_creature(card, def, game_state, card_db);

                bool passive_restricted = false;
                if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_ATTACK, card_db)) {
                    passive_restricted = true;
                }

                if (can_attack_player && !passive_restricted) {
                    CommandDef attack_player;
                    attack_player.type = CommandType::ATTACK_PLAYER;
                    attack_player.instance_id = card.instance_id;
                    attack_player.target_instance = -1;
                    attack_player.slot_index = static_cast<int>(i);
                    actions.push_back(attack_player);
                }

                if (can_attack_creature && !passive_restricted) {
                    for (size_t j = 0; j < opponent.battle_zone.size(); ++j) {
                        const auto& opp_card = opponent.battle_zone[j];
                        if (opp_card.is_tapped) {
                            if (card_db.count(opp_card.card_id)) {
                                const auto& opp_def = card_db.at(opp_card.card_id);
                                bool protected_by_jd = TargetUtils::is_protected_by_just_diver(opp_card, opp_def, game_state, active_player.id);
                                if (game_state.turn_number > opp_card.turn_played) protected_by_jd = false;
                                if (protected_by_jd) continue;
                            }

                            CommandDef attack_creature;
                            attack_creature.type = CommandType::ATTACK_CREATURE;
                            attack_creature.instance_id = card.instance_id;
                            attack_creature.target_instance = opp_card.instance_id;
                            attack_creature.slot_index = static_cast<int>(i);
                            attack_creature.target_slot_index = static_cast<int>(j);
                            actions.push_back(attack_creature);
                        }
                    }
                }
            }
        }

        CommandDef pass;
        pass.type = CommandType::PASS;
        actions.push_back(pass);

        return actions;
    }

    std::vector<CommandDef> BlockPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<CommandDef> actions;
        const auto& game_state = ctx.game_state;
        const auto& card_db = ctx.card_db;

        const Player& defender = game_state.players[1 - game_state.active_player_id];

        for (size_t i = 0; i < defender.battle_zone.size(); ++i) {
            const auto& card = defender.battle_zone[i];
            if (!card.is_tapped) {
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    if (TargetUtils::has_keyword_simple(game_state, card, def, "BLOCKER")) {
                        if (!PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_BLOCK, card_db)) {
                            CommandDef block;
                            block.type = CommandType::BLOCK;
                            block.instance_id = card.instance_id;
                            block.slot_index = static_cast<int>(i);
                            actions.push_back(block);
                        }
                    }
                }
            }
        }
        CommandDef pass;
        pass.type = CommandType::PASS;
        actions.push_back(pass);

        return actions;
    }

}
