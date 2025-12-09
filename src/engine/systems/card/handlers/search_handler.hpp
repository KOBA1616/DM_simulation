#pragma once
#include "../effect_system.hpp"
#include "../../../../core/game_state.hpp"
#include "../generic_card_system.hpp"
#include "../target_utils.hpp"
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
                 ed.condition = ConditionDef{"NONE", 0, ""};

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
                // Find controller
                PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                Player& controller = ctx.game_state.players[controller_id];

                int look = ctx.action.value1;
                // Variable Linking
                if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                    look = ctx.execution_vars[ctx.action.input_value_key];
                }
                if (look == 0) look = 1;

                std::vector<CardInstance> looked;
                for (int i = 0; i < look; ++i) {
                    if (controller.deck.empty()) break;
                    looked.push_back(controller.deck.back());
                    controller.deck.pop_back();
                }

                int chosen_idx = -1;
                for (size_t i = 0; i < looked.size(); ++i) {
                    const CardData* cd = CardRegistry::get_card_data(looked[i].card_id);
                    if (!cd) continue;

                    if (TargetUtils::is_valid_target(looked[i], *cd, ctx.action.filter, ctx.game_state, controller_id, controller_id)) {
                        chosen_idx = (int)i;
                        break;
                    }
                }

                if (chosen_idx != -1) {
                    controller.hand.push_back(looked[chosen_idx]);
                }

                for (int i = 0; i < (int)looked.size(); ++i) {
                    if (i == chosen_idx) continue;
                    controller.deck.insert(controller.deck.begin(), looked[i]);
                }
            }

            // SEND_TO_DECK_BOTTOM
             if (ctx.action.type == EffectActionType::SEND_TO_DECK_BOTTOM) {
                 if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                     EffectDef ed;
                     ed.trigger = TriggerType::NONE;
                     ed.condition = ConditionDef{"NONE", 0, ""};
                     ed.actions = { ctx.action };
                     GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 }
             }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;

             // SEARCH_DECK legacy/fallback path
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

             // SEND_TO_DECK_BOTTOM
             if (ctx.action.type == EffectActionType::SEND_TO_DECK_BOTTOM) {
                for (int tid : *ctx.targets) {
                    for (auto &p : ctx.game_state.players) {
                         // Check Hand
                         auto it = std::find_if(p.hand.begin(), p.hand.end(),
                             [tid](const CardInstance& c){ return c.instance_id == tid; });
                         if (it != p.hand.end()) {
                             p.deck.insert(p.deck.begin(), *it);
                             p.hand.erase(it);
                             continue;
                         }
                         // Check Battle Zone
                         auto bit = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                             [tid](const CardInstance& c){ return c.instance_id == tid; });
                         if (bit != p.battle_zone.end()) {
                             p.deck.insert(p.deck.begin(), *bit);
                             p.battle_zone.erase(bit);
                             continue;
                         }
                    }
                }
             }
        }
    };
}
