#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include <algorithm>
#include <random>

namespace dm::engine {

    class SearchHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // SEARCH_DECK
            if (ctx.action.type == EffectActionType::SEARCH_DECK) {
                 EffectDef ed;
                 ed.trigger = TriggerType::NONE;
                 ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};

                 ActionDef move_act;
                 move_act.type = EffectActionType::RETURN_TO_HAND;
                 if (ctx.action.destination_zone == "MANA_ZONE") {
                     move_act.type = EffectActionType::SEND_TO_MANA;
                 }

                 ActionDef shuffle_act;
                 shuffle_act.type = EffectActionType::SHUFFLE_DECK;

                 ed.actions = { move_act, shuffle_act };

                 ActionDef mod_action = ctx.action;
                 if (mod_action.filter.zones.empty()) {
                     mod_action.filter.zones = {"DECK"};
                 }
                 if (!mod_action.filter.owner.has_value()) {
                     mod_action.filter.owner = "SELF";
                 }
                 GenericCardSystem::select_targets(ctx.game_state, mod_action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            // SHUFFLE_DECK
            if (ctx.action.type == EffectActionType::SHUFFLE_DECK) {
                PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                Player& controller = ctx.game_state.players[controller_id];
                std::shuffle(controller.deck.begin(), controller.deck.end(), ctx.game_state.rng);
            }

            // SEARCH_DECK_BOTTOM
            if (ctx.action.type == EffectActionType::SEARCH_DECK_BOTTOM) {
                // Logic: Look at N cards, select M matching filter, add to hand, rest to bottom.

                // 1. Identify cards.
                PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                Player& controller = ctx.game_state.players[controller_id];

                int look = ctx.action.value1;
                if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                    look = ctx.execution_vars[ctx.action.input_value_key];
                }
                if (look == 0) look = 1;

                std::vector<CardID> buffer_card_ids;
                std::vector<int> buffer_instance_ids;

                // Move from Deck Bottom to Player's Effect Buffer
                // Deck front is bottom (implied by draw_handler using back)

                for (int i = 0; i < look; ++i) {
                    if (controller.deck.empty()) break;
                    // Move from Bottom (front) to Buffer
                    CardInstance c = controller.deck.front();
                    controller.deck.erase(controller.deck.begin());
                    controller.effect_buffer.push_back(c); // Use controller's buffer
                    buffer_card_ids.push_back(c.card_id);
                    buffer_instance_ids.push_back(c.instance_id);
                }

                // Now create a SELECT_TARGET action
                EffectDef ed;
                ed.trigger = TriggerType::NONE;
                ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};

                // Selected go to Hand
                ActionDef move_act;
                move_act.type = EffectActionType::RETURN_TO_HAND;

                ed.actions = { move_act };

                ActionDef selection_action = ctx.action;
                // Use SELECT_FROM_BUFFER as the action type for the selection phase
                selection_action.type = EffectActionType::SELECT_FROM_BUFFER;
                selection_action.filter.zones = {"EFFECT_BUFFER"};
                selection_action.filter.owner = "SELF";

                // IMPORTANT: We force the type to SEARCH_DECK_BOTTOM so we get the callback below
                // GenericCardSystem checks for generic handlers first? No, it uses the ActionType to dispatch?
                // GenericCardSystem::select_targets queues a PendingEffect.
                // The PendingEffect type will be `SELECT_TARGET` (usually) or the action type?
                // It sets `type = EffectActionType::SELECT_TARGET` (or similar) internally?
                // Actually `GenericCardSystem::select_targets` creates a `PendingEffect` with type `SELECT_TARGET`?
                // Wait, if it creates `SELECT_TARGET`, then `SearchHandler` won't be called for `resolve_with_targets`.
                // `GenericCardSystem` uses `resolve_effect_with_targets` which checks `ctx.action.type`.
                // `ctx.action` comes from the PendingEffect's stored action?
                // Yes.
                // So if we pass `selection_action` which has `SEARCH_DECK_BOTTOM` (if we didn't change it).

                // But we CHANGED `selection_action.type` to `SELECT_FROM_BUFFER` above!
                // So `SearchHandler` will NOT be called if `SELECT_FROM_BUFFER` doesn't map to it.
                // Does `SearchHandler` handle `SELECT_FROM_BUFFER`?
                // No.

                // We must use `SEARCH_DECK_BOTTOM` as the type.
                selection_action.type = EffectActionType::SEARCH_DECK_BOTTOM;

                // But `GenericCardSystem::select_targets` needs to know how to select.
                // It uses `scope` and `filter`.
                // If `scope` is TARGET_SELECT, it works.
                // Does `SEARCH_DECK_BOTTOM` imply TARGET_SELECT scope?
                // Yes, usually defined in JSON.
                // But we need to override the filter zones to EFFECT_BUFFER.

                selection_action.filter.zones = {"EFFECT_BUFFER"};
                selection_action.filter.owner = "SELF";

                GenericCardSystem::select_targets(ctx.game_state, selection_action, ctx.source_instance_id, ed, ctx.execution_vars);
            }

            // SEND_TO_DECK_BOTTOM
             if (ctx.action.type == EffectActionType::SEND_TO_DECK_BOTTOM) {
                 if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                     EffectDef ed;
                     ed.trigger = TriggerType::NONE;
                     ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                     ed.actions = { ctx.action };
                     GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 }
             }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;

             // SEARCH_DECK
             if (ctx.action.type == EffectActionType::SEARCH_DECK) {
                 Player& active = ctx.game_state.get_active_player();
                 for (int tid : *ctx.targets) {
                     auto it = std::find_if(active.deck.begin(), active.deck.end(),
                         [tid](const CardInstance& c){ return c.instance_id == tid; });
                     if (it != active.deck.end()) {
                         active.hand.push_back(*it);
                         active.deck.erase(it);
                     }
                 }
                 std::shuffle(active.deck.begin(), active.deck.end(), ctx.game_state.rng);
             }

             // SEARCH_DECK_BOTTOM (Cleanup Phase)
             if (ctx.action.type == EffectActionType::SEARCH_DECK_BOTTOM) {
                 PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                 Player& controller = ctx.game_state.players[controller_id];

                 // 1. Move Targets to Hand
                 for (int tid : *ctx.targets) {
                     // Check Buffer
                     auto it = std::find_if(controller.effect_buffer.begin(), controller.effect_buffer.end(),
                         [tid](const CardInstance& c){ return c.instance_id == tid; });
                     if (it != controller.effect_buffer.end()) {
                         controller.hand.push_back(*it);
                         controller.effect_buffer.erase(it);
                     }
                 }

                 // 2. Move Rest of Buffer to Deck Bottom
                 while (!controller.effect_buffer.empty()) {
                     controller.deck.insert(controller.deck.begin(), controller.effect_buffer.front());
                     controller.effect_buffer.erase(controller.effect_buffer.begin());
                 }
             }

             // SEND_TO_DECK_BOTTOM
             if (ctx.action.type == EffectActionType::SEND_TO_DECK_BOTTOM) {
                for (int tid : *ctx.targets) {
                    bool found = false;

                    // Check Buffer first
                    for (auto &p : ctx.game_state.players) {
                        auto bit_buf = std::find_if(p.effect_buffer.begin(), p.effect_buffer.end(),
                            [tid](const CardInstance& c){ return c.instance_id == tid; });
                        if (bit_buf != p.effect_buffer.end()) {
                            // Move from Buffer to Owner's Deck Bottom
                            p.deck.insert(p.deck.begin(), *bit_buf);
                            p.effect_buffer.erase(bit_buf);
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        for (auto &p : ctx.game_state.players) {
                             // Check Hand
                             auto it = std::find_if(p.hand.begin(), p.hand.end(),
                                 [tid](const CardInstance& c){ return c.instance_id == tid; });
                             if (it != p.hand.end()) {
                                 p.deck.insert(p.deck.begin(), *it);
                                 p.hand.erase(it);
                                 found = true;
                                 break;
                             }
                             // Check Battle Zone
                             auto bit = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                                 [tid](const CardInstance& c){ return c.instance_id == tid; });
                             if (bit != p.battle_zone.end()) {
                                 p.deck.insert(p.deck.begin(), *bit);
                                 p.battle_zone.erase(bit);
                                 found = true;
                                 break;
                             }
                        }
                    }
                }
             }
        }
    };
}
