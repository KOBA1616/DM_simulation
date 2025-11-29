#include "action_generator.hpp"
#include "../mana/mana_system.hpp"

namespace dm::engine {

    using namespace dm::core;

    std::vector<Action> ActionGenerator::generate_legal_actions(const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        std::vector<Action> actions;
        const Player& active_player = game_state.players[game_state.active_player_id];
        const Player& opponent = game_state.players[1 - game_state.active_player_id];

        switch (game_state.current_phase) {
            case Phase::MANA:
                // 1. Charge Mana (any card from hand)
                for (const auto& card : active_player.hand) {
                    Action action;
                    action.type = ActionType::MANA_CHARGE;
                    action.card_id = card.card_id;
                    action.source_instance_id = card.instance_id;
                    actions.push_back(action);
                }
                // 2. Pass (Always allowed in Mana phase? Spec says "5ターン目以降、チャージスキップ(PASS)可能 [Q26]")
                // But usually you can skip charge anytime. "Must charge 1 per turn" is not a rule, it's "Up to 1".
                // So Pass is always allowed.
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            case Phase::MAIN:
                // 1. Play Card
                for (const auto& card : active_player.hand) {
                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        if (ManaSystem::can_pay_cost(active_player, def, card_db)) {
                            Action action;
                            action.type = ActionType::PLAY_CARD;
                            action.card_id = card.card_id;
                            action.source_instance_id = card.instance_id;
                            actions.push_back(action);
                        }
                    }
                }
                // 2. Pass
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            case Phase::ATTACK:
                // 1. Attack with creatures
                for (const auto& card : active_player.battle_zone) {
                    if (!card.is_tapped && !card.summoning_sickness) {
                        // Attack Player
                        Action attack_player;
                        attack_player.type = ActionType::ATTACK_PLAYER;
                        attack_player.source_instance_id = card.instance_id;
                        attack_player.target_player = opponent.id;
                        actions.push_back(attack_player);

                        // Attack Tapped Creatures
                        for (const auto& opp_card : opponent.battle_zone) {
                            if (opp_card.is_tapped) {
                                Action attack_creature;
                                attack_creature.type = ActionType::ATTACK_CREATURE;
                                attack_creature.source_instance_id = card.instance_id;
                                attack_creature.target_instance_id = opp_card.instance_id;
                                actions.push_back(attack_creature);
                            }
                        }
                    }
                }
                // 2. Pass (End Attack Phase)
                {
                    Action pass;
                    pass.type = ActionType::PASS;
                    actions.push_back(pass);
                }
                break;

            default:
                break;
        }

        return actions;
    }

}
