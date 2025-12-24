#include "phase_strategies.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/systems/card/passive_effect_system.hpp"
#include "engine/cost_payment_system.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> ManaPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& game_state = ctx.game_state;
        const Player& active_player = game_state.players[game_state.active_player_id];

        for (size_t i = 0; i < active_player.hand.size(); ++i) {
            const auto& card = active_player.hand[i];
            Action action;
            action.type = PlayerIntent::MANA_CHARGE;
            action.card_id = card.card_id;
            action.source_instance_id = card.instance_id;
            action.slot_index = static_cast<int>(i);
            actions.push_back(action);
        }

        Action pass;
        pass.type = PlayerIntent::PASS;
        actions.push_back(pass);

        return actions;
    }

    std::vector<Action> MainPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& game_state = ctx.game_state;
        const auto& card_db = ctx.card_db;
        const Player& active_player = game_state.players[game_state.active_player_id];

        for (size_t i = 0; i < active_player.hand.size(); ++i) {
            const auto& card = active_player.hand[i];
            if (card_db.count(card.card_id)) {
                const auto& def = card_db.at(card.card_id);

                // Step 3-4: CANNOT_USE_SPELLS check (Creature check comes later if it's Twinpact)
                bool spell_restricted = false;
                if (def.type == CardType::SPELL) {
                    if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_USE_SPELLS, card_db)) {
                        spell_restricted = true;
                    }
                    // Lock by Cost check
                    if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::LOCK_SPELL_BY_COST, card_db)) {
                        spell_restricted = true;
                    }
                }

                // 1. Standard Play (Creature side if Twinpact)
                if (!spell_restricted && ManaSystem::can_pay_cost(game_state, active_player, def, card_db)) {
                    Action action;
                    action.type = PlayerIntent::DECLARE_PLAY;
                    action.card_id = card.card_id;
                    action.source_instance_id = card.instance_id;
                    action.slot_index = static_cast<int>(i);
                    actions.push_back(action);
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

                    if (!side_restricted && ManaSystem::can_pay_cost(game_state, active_player, spell_def, card_db)) {
                        Action action;
                        action.type = PlayerIntent::DECLARE_PLAY;
                        action.card_id = card.card_id; // Same ID? Or should we use spell_side ID?
                        // Usually Twinpact cards share ID but behave differently.
                        // We use `is_spell_side` flag.
                        action.source_instance_id = card.instance_id;
                        action.slot_index = static_cast<int>(i);
                        action.is_spell_side = true;
                        actions.push_back(action);
                    }
                }

                // 3. Active Cost Reductions (Phase 4)
                for (const auto& reduction : def.cost_reductions) {
                    if (reduction.type == ReductionType::ACTIVE_PAYMENT) {
                        int max_units = CostPaymentSystem::calculate_max_units(game_state, active_player.id, reduction, card_db);

                        // We assume "min_units" is usually 1 if we are invoking this cost.
                        // However, we should iterate all possible payments (1..max_units)
                        // For now, strict Hyper Energy loop: taps 1..max
                        for (int units = 1; units <= max_units; ++units) {
                            if (reduction.max_units != -1 && units > reduction.max_units) break;

                            int reduction_val = units * reduction.reduction_amount;
                            // Calculate adjusted cost including passive modifiers first
                            int adjusted_cost = ManaSystem::get_adjusted_cost(game_state, active_player, def);
                            int effective_cost = std::max(reduction.min_mana_cost, adjusted_cost - reduction_val);

                            int available_mana = ManaSystem::get_usable_mana_count(game_state, active_player.id, def.civilizations, card_db);

                            if (available_mana >= effective_cost) {
                                Action action;
                                action.type = PlayerIntent::DECLARE_PLAY;
                                action.card_id = card.card_id;
                                action.source_instance_id = card.instance_id;
                                action.slot_index = static_cast<int>(i);
                                action.target_slot_index = units; // Use target_slot_index for payment amount
                                action.target_player = 254; // Legacy flag for "Special/Active Payment Play"
                                // We might want to pass WHICH reduction is used if multiple exist.
                                // For now, assume 1 active reduction per card or take first valid.
                                // If needed, we can use value1 = reduction_index.
                                // But CostPaymentSystem::calculate_max_units handles the logic per reduction.
                                actions.push_back(action);
                            }
                        }
                    }
                }
            }
        }

        Action pass;
        pass.type = PlayerIntent::PASS;
        actions.push_back(pass);

        return actions;
    }

    std::vector<Action> AttackPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
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

                // Passive check for general attacking restriction
                bool passive_restricted = false;
                if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_ATTACK, card_db)) {
                    passive_restricted = true;
                }

                // Attack Player
                if (can_attack_player && !passive_restricted) {
                    Action attack_player;
                    attack_player.type = PlayerIntent::ATTACK_PLAYER;
                    attack_player.source_instance_id = card.instance_id;
                    attack_player.slot_index = static_cast<int>(i);
                    attack_player.target_player = opponent.id;
                    actions.push_back(attack_player);
                }

                // Attack Creature
                if (can_attack_creature && !passive_restricted) {
                    for (size_t j = 0; j < opponent.battle_zone.size(); ++j) {
                        const auto& opp_card = opponent.battle_zone[j];
                        if (opp_card.is_tapped) {
                            if (card_db.count(opp_card.card_id)) {
                                const auto& opp_def = card_db.at(opp_card.card_id);
                                bool protected_by_jd = TargetUtils::is_protected_by_just_diver(opp_card, opp_def, game_state, active_player.id);
                                // Just Diver protection only lasts the turn it entered
                                if (game_state.turn_number > opp_card.turn_played) protected_by_jd = false;
                                if (protected_by_jd) {
                                    continue;
                                }
                            }

                            Action attack_creature;
                            attack_creature.type = PlayerIntent::ATTACK_CREATURE;
                            attack_creature.source_instance_id = card.instance_id;
                            attack_creature.slot_index = static_cast<int>(i);
                            attack_creature.target_instance_id = opp_card.instance_id;
                            attack_creature.target_slot_index = static_cast<int>(j);
                            actions.push_back(attack_creature);
                        }
                    }
                }
            }
        }

        Action pass;
        pass.type = PlayerIntent::PASS;
        actions.push_back(pass);

        return actions;
    }

    std::vector<Action> BlockPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& game_state = ctx.game_state;
        const auto& card_db = ctx.card_db;

        const Player& defender = game_state.players[1 - game_state.active_player_id];

        for (size_t i = 0; i < defender.battle_zone.size(); ++i) {
            const auto& card = defender.battle_zone[i];
            if (!card.is_tapped) {
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    if (TargetUtils::has_keyword_simple(game_state, card, def, "BLOCKER")) {
                        // Step 3-4: CANNOT_BLOCK check
                        if (!PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_BLOCK, card_db)) {
                            Action block;
                            block.type = PlayerIntent::BLOCK;
                            block.source_instance_id = card.instance_id;
                            block.slot_index = static_cast<int>(i);
                            actions.push_back(block);
                        }
                    }
                }
            }
        }
        Action pass;
        pass.type = PlayerIntent::PASS;
        actions.push_back(pass);
        return actions;
    }

}
