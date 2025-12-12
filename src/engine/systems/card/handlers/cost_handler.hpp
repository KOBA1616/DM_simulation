#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/effects/effect_resolver.hpp"

namespace dm::engine {

    class CostHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (ctx.action.type == EffectActionType::COST_REFERENCE) {
                if (ctx.action.str_val == "FINISH_HYPER_ENERGY") {
                     if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                         EffectDef ed;
                         ed.trigger = TriggerType::NONE;
                         ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                         ed.actions = { ctx.action };
                         GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                     }
                }
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.targets) return;

             if (ctx.action.type == EffectActionType::COST_REFERENCE && ctx.action.str_val == "FINISH_HYPER_ENERGY") {
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

                 PlayerID controller = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                 EffectResolver::resolve_play_from_stack(ctx.game_state, ctx.source_instance_id, reduction, SpawnSource::HAND_SUMMON, controller, ctx.card_db);
             }
        }
    };
}
