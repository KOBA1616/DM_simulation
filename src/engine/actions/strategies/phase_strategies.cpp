#include "phase_strategies.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/systems/card/passive_effect_system.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> ManaPhaseStrategy::generate(const ActionGenContext& ctx) {
        std::vector<Action> actions;
        const auto& game_state = ctx.game_state;
        const Player& active_player = game_state.players[game_state.active_player_id];

        for (size_t i = 0; i < active_player.hand.size(); ++i) {
            const auto& card = active_player.hand[i];
            Action action;
            action.type = ActionType::MOVE_CARD;
            action.card_id = card.card_id;
            action.source_instance_id = card.instance_id;
            action.slot_index = static_cast<int>(i);
            actions.push_back(action);
        }

        Action pass;
        pass.type = ActionType::PASS;
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
                    action.type = ActionType::DECLARE_PLAY;
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
                        action.type = ActionType::DECLARE_PLAY;
                        action.card_id = card.card_id; // Same ID? Or should we use spell_side ID?
                        // Usually Twinpact cards share ID but behave differently.
                        // We use `is_spell_side` flag.
                        action.source_instance_id = card.instance_id;
                        action.slot_index = static_cast<int>(i);
                        action.is_spell_side = true;
                        actions.push_back(action);
                    }
                }

                if (def.keywords.hyper_energy) {
                    int untapped_creatures = 0;
                    for (const auto& c : active_player.battle_zone) {
                        if (!c.is_tapped && !c.summoning_sickness) untapped_creatures++;
                    }

                    for (int taps = 1; taps <= untapped_creatures; ++taps) {
                        int reduction = taps * 2;
                        int effective_cost = std::max(0, def.cost - reduction);

                        int available_mana = 0;
                        for(const auto& m : active_player.mana_zone) if(!m.is_tapped) available_mana++;

                        if (available_mana >= effective_cost) {
                            Action action;
                            action.type = ActionType::DECLARE_PLAY;
                            action.card_id = card.card_id;
                            action.source_instance_id = card.instance_id;
                            action.slot_index = static_cast<int>(i);
                            action.target_slot_index = taps;
                            action.target_player = 254;
                            actions.push_back(action);
                        }
                    }
                }
            }
        }

        Action pass;
        pass.type = ActionType::PASS;
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

            bool can_attack = !card.is_tapped;
            if (can_attack) {
                if (card.summoning_sickness) {
                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        bool has_sa = TargetUtils::has_keyword_simple(game_state, card, def, "SPEED_ATTACKER");
                        bool is_evo = TargetUtils::has_keyword_simple(game_state, card, def, "EVOLUTION"); // EVOLUTION is a keyword? Or Type? Or Prop?
                        // EVOLUTION as keyword flag in CardKeywords is true.

                        if (!has_sa && !is_evo) {
                            can_attack = false;
                        }
                    } else {
                        can_attack = false;
                    }
                }
            }

            // Step 3-4: CANNOT_ATTACK check
            if (can_attack) {
                if (PassiveEffectSystem::instance().check_restriction(game_state, card, PassiveType::CANNOT_ATTACK, card_db)) {
                    can_attack = false;
                }
            }

            if (can_attack) {
                Action attack_player;
                attack_player.type = ActionType::ATTACK_PLAYER;
                attack_player.source_instance_id = card.instance_id;
                attack_player.slot_index = static_cast<int>(i);
                attack_player.target_player = opponent.id;
                actions.push_back(attack_player);

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
                        attack_creature.type = ActionType::ATTACK_CREATURE;
                        attack_creature.source_instance_id = card.instance_id;
                        attack_creature.slot_index = static_cast<int>(i);
                        attack_creature.target_instance_id = opp_card.instance_id;
                        attack_creature.target_slot_index = static_cast<int>(j);
                        actions.push_back(attack_creature);
                    }
                }
            }
        }

        Action pass;
        pass.type = ActionType::PASS;
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
                            block.type = ActionType::BLOCK;
                            block.source_instance_id = card.instance_id;
                            block.slot_index = static_cast<int>(i);
                            actions.push_back(block);
                        }
                    }
                }
            }
        }
        Action pass;
        pass.type = ActionType::PASS;
        actions.push_back(pass);
        return actions;
    }

}
