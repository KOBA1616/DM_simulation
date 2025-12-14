#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/utils/tap_in_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/card_def.hpp"

namespace dm::engine {

    class ManaChargeHandler : public IActionHandler {
    public:
        // Case 1: ADD_MANA (Top of deck to Mana)
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;

             PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
             Player& controller = ctx.game_state.players[controller_id];

             int count = ctx.action.value1;
             if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                count = ctx.execution_vars[ctx.action.input_value_key];
             }
             if (count == 0) count = 1;

             for (int i = 0; i < count; ++i) {
                if (controller.deck.empty()) break;
                CardInstance c = controller.deck.back();
                controller.deck.pop_back();
                c.is_tapped = false;

                // Tap-in Logic
                if (ctx.card_db.count(c.card_id)) {
                    TapInUtils::apply_tap_in_rule(c, ctx.card_db.at(c.card_id));
                }

                controller.mana_zone.push_back(c);
             }
        }

        // Case 2: SEND_TO_MANA (Target to Mana)
        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.targets) return;

            for (int tid : *ctx.targets) {
                // Find and move logic
                for (auto &p : ctx.game_state.players) {
                     auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                     if (it != p.battle_zone.end()) {
                         // Hierarchy Cleanup
                         ZoneUtils::on_leave_battle_zone(ctx.game_state, *it);

                         CardInstance moved = *it;
                         p.battle_zone.erase(it);
                         moved.is_tapped = false;

                         // Tap-in Logic
                         if (ctx.card_db.count(moved.card_id)) {
                             TapInUtils::apply_tap_in_rule(moved, ctx.card_db.at(moved.card_id));
                         }

                         p.mana_zone.push_back(moved);

                         GenericCardSystem::check_mega_last_burst(ctx.game_state, moved, ctx.card_db);
                         break;
                     }
                     // Hand -> Mana
                     auto it_hand = std::find_if(p.hand.begin(), p.hand.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                     if (it_hand != p.hand.end()) {
                         CardInstance moved = *it_hand;
                         p.hand.erase(it_hand);
                         moved.is_tapped = false;

                         // Tap-in
                         if (ctx.card_db.count(moved.card_id)) {
                             TapInUtils::apply_tap_in_rule(moved, ctx.card_db.at(moved.card_id));
                         }
                         p.mana_zone.push_back(moved);
                         break;
                     }
                }
            }
        }
    };
}
