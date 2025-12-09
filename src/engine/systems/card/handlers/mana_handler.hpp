#pragma once
#include "../effect_system.hpp"
#include "../../../../core/game_state.hpp"
#include "../../../../utils/tap_in_utils.hpp"
#include "../../../../utils/zone_utils.hpp"
#include "../../../../core/card_def.hpp"

namespace dm::engine {

    class ManaChargeHandler : public IActionHandler {
    public:
        // Case 1: ADD_MANA (Top of deck to Mana)
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
             using namespace dm::core;

             PlayerID controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
             Player& controller = game_state.players[controller_id];

             int count = action.value1;
             if (!action.input_value_key.empty() && execution_context.count(action.input_value_key)) {
                count = execution_context[action.input_value_key];
             }
             if (count == 0) count = 1;

             for (int i = 0; i < count; ++i) {
                if (controller.deck.empty()) break;
                CardInstance c = controller.deck.back();
                controller.deck.pop_back();
                c.is_tapped = false;

                // Tap-in Logic
                const CardData* data = CardRegistry::get_card_data(c.card_id);
                if (data) {
                    if (data->civilizations.size() > 1) {
                         bool untap_in = false;
                         if (data->keywords.has_value() && data->keywords.value().count("untap_in")) {
                             untap_in = data->keywords.value().at("untap_in");
                         }
                         if (!untap_in) {
                             c.is_tapped = true;
                         }
                    }
                }

                controller.mana_zone.push_back(c);
             }
        }

        // Case 2: SEND_TO_MANA (Target to Mana)
        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) override {
            using namespace dm::core;

            for (int tid : targets) {
                // Find and move logic
                // Using explicit loops for now as per DestroyHandler style
                for (auto &p : game_state.players) {
                     auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                     if (it != p.battle_zone.end()) {
                         // Hierarchy Cleanup
                         ZoneUtils::on_leave_battle_zone(game_state, *it);

                         CardInstance moved = *it;
                         p.battle_zone.erase(it);
                         moved.is_tapped = false;

                         // Tap-in Logic
                         if (card_db.count(moved.card_id)) {
                             TapInUtils::apply_tap_in_rule(moved, card_db.at(moved.card_id));
                         }

                         p.mana_zone.push_back(moved);
                         break; // Assuming unique ID
                     }
                     // Could be other zones depending on filter, but BattleZone is main target for SEND_TO_MANA
                     // Hand -> Mana
                     auto it_hand = std::find_if(p.hand.begin(), p.hand.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                     if (it_hand != p.hand.end()) {
                         // Hierarchy N/A for hand
                         CardInstance moved = *it_hand;
                         p.hand.erase(it_hand);
                         moved.is_tapped = false;

                         // Tap-in
                         if (card_db.count(moved.card_id)) {
                             TapInUtils::apply_tap_in_rule(moved, card_db.at(moved.card_id));
                         }
                         p.mana_zone.push_back(moved);
                         break;
                     }
                }
            }
        }
    };
}
