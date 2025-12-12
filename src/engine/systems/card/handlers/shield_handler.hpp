#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include <algorithm>
#include <vector>
#include <set>

namespace dm::engine {

    class ShieldHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
            Player& controller = ctx.game_state.players[controller_id];

            if (ctx.action.type == EffectActionType::ADD_SHIELD) {
                std::vector<CardInstance>* source = &controller.deck;
                if (ctx.action.source_zone == "HAND") source = &controller.hand;
                else if (ctx.action.source_zone == "GRAVEYARD") source = &controller.graveyard;

                int count = ctx.action.value1;
                if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                    count = ctx.execution_vars[ctx.action.input_value_key];
                }
                if (count == 0) count = 1;

                int added_count = 0;
                for (int i = 0; i < count; ++i) {
                    if (!source->empty()) {
                        CardInstance c = source->back();
                        source->pop_back();
                        c.is_face_down = true;
                        controller.shield_zone.push_back(c);
                        added_count++;
                    }
                }
                if (!ctx.action.output_value_key.empty()) {
                    ctx.execution_vars[ctx.action.output_value_key] = added_count;
                }
            } else if (ctx.action.type == EffectActionType::SEND_SHIELD_TO_GRAVE) {
                if (ctx.action.scope != TargetScope::TARGET_SELECT && ctx.action.target_choice != "SELECT") {
                     if (!controller.shield_zone.empty()) {
                        CardInstance c = controller.shield_zone.back();
                        controller.shield_zone.pop_back();
                        controller.graveyard.push_back(c);
                    }
                } else {
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

             if (ctx.action.type == EffectActionType::SEND_SHIELD_TO_GRAVE) {

                 std::set<int> target_ids(ctx.targets->begin(), ctx.targets->end());
                 std::vector<int> final_targets;

                 if (ctx.action.inverse_target) {
                     std::vector<int> players_to_check;
                     if (ctx.action.filter.owner.has_value()) {
                        std::string req = ctx.action.filter.owner.value();
                        PlayerID controller = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                        if (req == "SELF") players_to_check.push_back(controller);
                        else if (req == "OPPONENT") players_to_check.push_back(1 - controller);
                        else if (req == "BOTH") { players_to_check.push_back(controller); players_to_check.push_back(1 - controller); }
                     } else {
                         players_to_check.push_back(0);
                         players_to_check.push_back(1);
                     }

                     for (int pid : players_to_check) {
                         const auto& p = ctx.game_state.players[pid];
                         for (const auto& s : p.shield_zone) {
                             if (target_ids.find(s.instance_id) == target_ids.end()) {
                                 final_targets.push_back(s.instance_id);
                             }
                         }
                     }
                 } else {
                     final_targets = *ctx.targets;
                 }

                 for (int tid : final_targets) {
                    for (auto &p : ctx.game_state.players) {
                         auto it = std::find_if(p.shield_zone.begin(), p.shield_zone.end(),
                            [tid](const CardInstance& c){ return c.instance_id == tid; });
                         if (it != p.shield_zone.end()) {
                             p.graveyard.push_back(*it);
                             p.shield_zone.erase(it);
                             break;
                         }
                    }
                }
             }
        }
    };
}
