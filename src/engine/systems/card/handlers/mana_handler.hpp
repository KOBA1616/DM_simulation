#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/utils/tap_in_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"

namespace dm::engine {

    class ManaChargeHandler : public IActionHandler {
    public:
        // Case 1: ADD_MANA (Top of deck to Mana)
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;

             // Check if it's SEND_TO_MANA (Auto Mode) or ADD_MANA (Deck Mode)
             // ADD_MANA usually has count > 0.
             // SEND_TO_MANA without targets might be Auto (e.g. "Put all creatures into mana")
             // We can distinguish by Action Type?
             // But GenericCardSystem delegates both to here if registered?
             // Currently only ADD_MANA is registered. We will register SEND_TO_MANA too.

             if (ctx.action.type == EffectActionType::SEND_TO_MANA) {
                 // Auto-Send Logic (e.g. "Put all creatures into mana")
                 if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                     EffectDef ed;
                     ed.actions = { ctx.action };
                     EffectSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                     return;
                 }

                 // Auto-Loop
                 PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                 std::vector<std::pair<PlayerID, Zone>> zones_to_check;
                 if (ctx.action.filter.zones.empty()) {
                     zones_to_check.push_back({0, Zone::BATTLE});
                     zones_to_check.push_back({1, Zone::BATTLE});
                 } else {
                     for (const auto& z : ctx.action.filter.zones) {
                         if (z == "BATTLE_ZONE") {
                             zones_to_check.push_back({0, Zone::BATTLE});
                             zones_to_check.push_back({1, Zone::BATTLE});
                         }
                         // Hand? Graveyard?
                         if (z == "HAND") {
                             zones_to_check.push_back({0, Zone::HAND});
                             zones_to_check.push_back({1, Zone::HAND});
                         }
                         if (z == "GRAVEYARD") {
                             zones_to_check.push_back({0, Zone::GRAVEYARD});
                             zones_to_check.push_back({1, Zone::GRAVEYARD});
                         }
                     }
                 }

                 std::vector<int> targets_to_send;
                 for (const auto& [pid, zone] : zones_to_check) {
                    Player& p = ctx.game_state.players[pid];
                    const std::vector<CardInstance>* card_list = nullptr;
                    if (zone == Zone::BATTLE) card_list = &p.battle_zone;
                    else if (zone == Zone::HAND) card_list = &p.hand;
                    else if (zone == Zone::GRAVEYARD) card_list = &p.graveyard;

                    if (!card_list) continue;

                    for (const auto& card : *card_list) {
                        if (!ctx.card_db.count(card.card_id)) continue;
                        const auto& def = ctx.card_db.at(card.card_id);

                        if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                             if (pid != controller_id && zone == Zone::BATTLE) {
                                  if (TargetUtils::is_protected_by_just_diver(card, def, ctx.game_state, controller_id)) continue;
                             }
                             targets_to_send.push_back(card.instance_id);
                        }
                    }
                 }

                 for (int tid : targets_to_send) {
                     // Move Logic (Similar to resolve_with_targets)
                    for (auto &p : ctx.game_state.players) {
                         auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                            [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                         if (it != p.battle_zone.end()) {
                             ZoneUtils::on_leave_battle_zone(ctx.game_state, *it);
                             CardInstance moved = *it;
                             p.battle_zone.erase(it);
                             moved.is_tapped = false;
                             if (ctx.card_db.count(moved.card_id)) TapInUtils::apply_tap_in_rule(moved, ctx.card_db.at(moved.card_id));
                             p.mana_zone.push_back(moved);
                             EffectSystem::instance().check_mega_last_burst(ctx.game_state, moved, ctx.card_db);
                             break;
                         }
                         auto it_hand = std::find_if(p.hand.begin(), p.hand.end(),
                            [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                         if (it_hand != p.hand.end()) {
                             CardInstance moved = *it_hand;
                             p.hand.erase(it_hand);
                             moved.is_tapped = false;
                             if (ctx.card_db.count(moved.card_id)) TapInUtils::apply_tap_in_rule(moved, ctx.card_db.at(moved.card_id));
                             p.mana_zone.push_back(moved);
                             break;
                         }
                         auto it_grave = std::find_if(p.graveyard.begin(), p.graveyard.end(),
                            [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                         if (it_grave != p.graveyard.end()) {
                             CardInstance moved = *it_grave;
                             p.graveyard.erase(it_grave);
                             moved.is_tapped = false;
                             if (ctx.card_db.count(moved.card_id)) TapInUtils::apply_tap_in_rule(moved, ctx.card_db.at(moved.card_id));
                             p.mana_zone.push_back(moved);
                             break;
                         }
                    }
                 }
                 return;
             }

             // ADD_MANA Logic (Deck -> Mana)
             PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
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

                         EffectSystem::instance().check_mega_last_burst(ctx.game_state, moved, ctx.card_db);
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
                     // Graveyard -> Mana
                     auto it_grave = std::find_if(p.graveyard.begin(), p.graveyard.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                     if (it_grave != p.graveyard.end()) {
                         CardInstance moved = *it_grave;
                         p.graveyard.erase(it_grave);
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

        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            if (ctx.action.type == EffectActionType::SEND_TO_MANA) {
                Instruction move(InstructionOp::MOVE);
                move.args["to"] = "MANA";

                if (!ctx.action.input_value_key.empty()) {
                    move.args["target"] = "$" + ctx.action.input_value_key;
                } else if (ctx.action.scope == TargetScope::TARGET_SELECT) {
                     move.args["target"] = "$selection";
                } else {
                     Instruction select(InstructionOp::SELECT);
                     select.args["filter"] = ctx.action.filter;
                     select.args["out"] = "$auto_mana_selection";
                     select.args["count"] = 999;

                     ctx.instruction_buffer->push_back(select);
                     move.args["target"] = "$auto_mana_selection";
                }

                ctx.instruction_buffer->push_back(move);
            }
            else { // ADD_MANA
                int count = ctx.action.value1;
                if (count == 0) count = 1;

                Instruction move(InstructionOp::MOVE);
                move.args["to"] = "MANA";
                move.args["target"] = "DECK_TOP";
                move.args["count"] = count;

                ctx.instruction_buffer->push_back(move);
            }
        }
    };
}
