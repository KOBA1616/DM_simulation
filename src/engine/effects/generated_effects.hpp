#pragma once
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include <vector>
#include <algorithm>

namespace dm::engine {

    class GeneratedEffects {
    public:
        static void resolve(dm::core::GameState& game_state, const dm::core::PendingEffect& effect, int card_id) {
            auto& controller = game_state.players[effect.controller];
            auto& opponent = game_state.players[1 - effect.controller];

            switch (card_id) {
                case 1: // Bronze-Arm Tribe
                    if (!controller.deck.empty()) {
                        auto card = controller.deck.back();
                        controller.deck.pop_back();
                        card.is_tapped = false;
                        controller.mana_zone.push_back(card);
                    }
                    break;
                case 2: // Aqua Hulcus
                    for (int i = 0; i < 1; ++i) {
                        if (!controller.deck.empty()) {
                            auto card = controller.deck.back();
                            controller.deck.pop_back();
                            controller.hand.push_back(card);
                        }
                    }
                    break;
                case 3: // Holy Awe
                    for (auto& c : opponent.battle_zone) {
                        c.is_tapped = true;
                    }
                    break;
                case 5: // Terror Pit
                    if (!effect.target_instance_ids.empty()) {
                        int target_id = effect.target_instance_ids[0];
                        auto it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                            [target_id](const dm::core::CardInstance& c) { return c.instance_id == target_id; });
                        if (it != opponent.battle_zone.end()) {
                            opponent.graveyard.push_back(*it);
                            opponent.battle_zone.erase(it);
                        }
                    }
                    break;
                case 6: // Spiral Gate
                    if (!effect.target_instance_ids.empty()) {
                        int target_id = effect.target_instance_ids[0];
                        // Check opponent
                        auto it_opp = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                            [target_id](const dm::core::CardInstance& c) { return c.instance_id == target_id; });
                        if (it_opp != opponent.battle_zone.end()) {
                            opponent.hand.push_back(*it_opp);
                            opponent.battle_zone.erase(it_opp);
                        } else {
                            // Check self
                            auto it_self = std::find_if(controller.battle_zone.begin(), controller.battle_zone.end(),
                                [target_id](const dm::core::CardInstance& c) { return c.instance_id == target_id; });
                            if (it_self != controller.battle_zone.end()) {
                                controller.hand.push_back(*it_self);
                                controller.battle_zone.erase(it_self);
                            }
                        }
                    }
                    break;
                default:
                    break;
            }
        }
    };
}
