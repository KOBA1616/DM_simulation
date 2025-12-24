#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/game_logic_system.hpp"
#include <set>
#include <stdexcept>

namespace dm::engine {

    class CostHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (ctx.action.type == EffectPrimitive::COST_REFERENCE) {
                if (ctx.action.str_val == "FINISH_HYPER_ENERGY") {
                     if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                         EffectDef ed;
                         ed.trigger = TriggerType::NONE;
                         ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                         ed.actions = { ctx.action };
                         EffectSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                     }
                }
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             using namespace dm::engine::systems;

             if (!ctx.targets) return;

             if (ctx.action.type == EffectPrimitive::COST_REFERENCE && ctx.action.str_val == "FINISH_HYPER_ENERGY") {
                 std::set<int> seen_costs;
                 bool valid_costs = true;

                 for (int tid : *ctx.targets) {
                    int card_id = -1;
                    bool found = false;
                    for (const auto &p : ctx.game_state.players) {
                        for (const auto& c : p.battle_zone) {
                            if (c.instance_id == tid) {
                                card_id = c.card_id;
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }

                    if (found && card_id != -1) {
                        if (ctx.card_db.count(card_id)) {
                             int cost = ctx.card_db.at(card_id).cost;
                             if (seen_costs.count(cost)) {
                                 valid_costs = false;
                                 break;
                             }
                             seen_costs.insert(cost);
                        }
                    }
                 }

                 if (!valid_costs) {
                     throw std::runtime_error("Hyper Energy requires creatures with different costs.");
                 }

                 for (int tid : *ctx.targets) {
                     for (auto &p : ctx.game_state.players) {
                          auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                             [tid](const CardInstance& c){ return c.instance_id == tid; });
                          if (it != p.battle_zone.end()) {
                              it->is_tapped = true;
                          }
                     }
                 }
                 int taps = ctx.action.value1;
                 if (taps == 0) taps = (int)ctx.targets->size();

                 int reduction = taps * 2;

                 PlayerID controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                 GameLogicSystem::resolve_play_from_stack(ctx.game_state, ctx.source_instance_id, reduction, SpawnSource::HAND_SUMMON, controller, ctx.card_db);
             }
        }
    };
}
