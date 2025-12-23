import json
import os

def generate_cpp(json_path, output_path):
    with open(json_path, 'r') as f:
        cards = json.load(f)

    code = """#pragma once
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
"""

    for card in cards:
        cid = card['id']
        code += f"                case {cid}: // {card['name']}\n"
        
        for eff in card.get('effects', []):
            etype = eff['type']
            
            if etype == "mana_charge":
                if eff.get('source') == "deck_top":
                    code += """                    if (!controller.deck.empty()) {
                        auto card = controller.deck.back();
                        controller.deck.pop_back();
                        card.is_tapped = false;
                        controller.mana_zone.push_back(card);
                    }
"""
            elif etype == "draw_card":
                amount = eff.get('amount', 1)
                code += f"""                    for (int i = 0; i < {amount}; ++i) {{
                        if (!controller.deck.empty()) {{
                            auto card = controller.deck.back();
                            controller.deck.pop_back();
                            controller.hand.push_back(card);
                        }}
                    }}
"""
            elif etype == "tap_all":
                if eff.get('target') == "opponent_creatures":
                    code += """                    for (auto& c : opponent.battle_zone) {
                        c.is_tapped = true;
                    }
"""
            elif etype == "destroy":
                if eff.get('target') == "select_opponent_creature":
                    code += """                    if (!effect.target_instance_ids.empty()) {
                        int target_id = effect.target_instance_ids[0];
                        auto it = std::find_if(opponent.battle_zone.begin(), opponent.battle_zone.end(),
                            [target_id](const dm::core::CardInstance& c) { return c.instance_id == target_id; });
                        if (it != opponent.battle_zone.end()) {
                            opponent.graveyard.push_back(*it);
                            opponent.battle_zone.erase(it);
                        }
                    }
"""
            elif etype == "bounce":
                if eff.get('target') == "select_creature":
                    code += """                    if (!effect.target_instance_ids.empty()) {
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
"""
        
        code += "                    break;\n"

    code += """                default:
                    break;
            }
        }
    };
}
"""

    with open(output_path, 'w') as f:
        f.write(code)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_cpp("data/card_effects.json", "src/engine/effects/generated_effects.hpp")
