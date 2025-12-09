#pragma once
#include "../effect_system.hpp"
#include "../../../../core/game_state.hpp"
#include "../generic_card_system.hpp"
#include "../../../utils/zone_utils.hpp"
#include <algorithm>

namespace dm::engine {

    class ReturnToHandHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
             if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.actions = { ctx.action };
                 GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             if (!ctx.targets) return;
             for (int tid : *ctx.targets) {
                for (auto &p : ctx.game_state.players) {
                    auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it != p.battle_zone.end()) {
                        // Cleanup hierarchy
                        ZoneUtils::on_leave_battle_zone(ctx.game_state, *it);

                        p.hand.push_back(*it);
                        p.battle_zone.erase(it);
                        break;
                    }
                    auto it_mana = std::find_if(p.mana_zone.begin(), p.mana_zone.end(),
                        [tid](const dm::core::CardInstance& c){ return c.instance_id == tid; });
                    if (it_mana != p.mana_zone.end()) {
                         p.hand.push_back(*it_mana);
                         p.mana_zone.erase(it_mana);
                         break;
                    }
                }
            }
        }
    };
}
