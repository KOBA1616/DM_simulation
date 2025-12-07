#include "phase_strategies.hpp"
#include "../../card_system/target_utils.hpp"
#include "../../mana/mana_system.hpp"

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

                if (ManaSystem::can_pay_cost(game_state, active_player, def, card_db)) {
                    Action action;
                    action.type = ActionType::DECLARE_PLAY;
                    action.card_id = card.card_id;
                    action.source_instance_id = card.instance_id;
                    action.slot_index = static_cast<int>(i);
                    actions.push_back(action);
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
                        if (!def.keywords.speed_attacker && !def.keywords.evolution) {
                            can_attack = false;
                        }
                    } else {
                        can_attack = false;
                    }
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
                            if (TargetUtils::is_protected_by_just_diver(opp_card, opp_def, game_state, active_player.id)) {
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

        // In Block Phase, NAP acts.
        // The context.active_player_id is set to game_state.active_player_id (Turn Player).
        // But the strategies should know who is acting.
        // The implementation in ActionGenerator handled this:
        // if (game_state.current_phase == Phase::BLOCK) { const Player& defender = opponent; ... }

        const Player& defender = game_state.players[1 - game_state.active_player_id];

        for (size_t i = 0; i < defender.battle_zone.size(); ++i) {
            const auto& card = defender.battle_zone[i];
            if (!card.is_tapped) {
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    if (def.keywords.blocker) {
                        Action block;
                        block.type = ActionType::BLOCK;
                        block.source_instance_id = card.instance_id;
                        block.slot_index = static_cast<int>(i);
                        actions.push_back(block);
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
