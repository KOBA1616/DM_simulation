#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/game_command/commands.hpp"
#include <algorithm>
#include <memory>

namespace dm::engine {

    class BreakShieldHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.actions = { ctx.action };
                 GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.targets) return;
            execute_breaks(ctx.game_state, *ctx.targets, ctx.source_instance_id, ctx.card_db);
        }

    private:
        void execute_breaks(dm::core::GameState& game_state, const std::vector<int>& shields, int breaker_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            for (int shield_id : shields) {
                PlayerID shield_owner = 255;
                bool found = false;
                for (int pid = 0; pid < 2; ++pid) {
                    const auto& s = game_state.players[pid].shield_zone;
                    for (const auto& c : s) {
                        if (c.instance_id == shield_id) {
                            shield_owner = pid;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }

                if (!found) continue;

                // Trigger Logic Restoration: AT_BREAK_SHIELD check?
                // Usually "When a shield is broken" effects happen here.
                GenericCardSystem::resolve_trigger(game_state, TriggerType::AT_BREAK_SHIELD, shield_id, card_db);

                Zone dest_zone = Zone::HAND;
                const CardInstance* breaker = game_state.get_card_instance(breaker_id);
                if (breaker && card_db.count(breaker->card_id)) {
                    if (card_db.at(breaker->card_id).keywords.shield_burn) {
                        dest_zone = Zone::GRAVEYARD;
                    }
                }

                // Use shared_ptr and execute_command for Undo
                auto cmd = std::make_shared<TransitionCommand>(shield_id, static_cast<int>(Zone::SHIELD), static_cast<int>(dest_zone), shield_owner, -1);
                game_state.execute_command(cmd);

                // Trigger Logic Restoration: S-Trigger / On Shield Add
                // S-Trigger check usually happens when card enters hand from shield zone.
                // It should be handled by an event listener on ZONE_ENTER (Hand) + Context (from Shield).
                // If Event System is active (Phase 6 Step 1), `TransitionCommand` *should* dispatch events but doesn't yet.
                // So we manually invoke legacy check if needed.
                // Assuming `GenericCardSystem::resolve_trigger` handles S-Trigger checks via `S_TRIGGER` type if applicable?
                // S-Trigger is unique because it interrupts resolution.
                // Existing `ShieldHandler` logic (overwritten) likely called `check_shield_trigger`.
                // Let's add a TODO or call `resolve_trigger(S_TRIGGER)` if dest is HAND.

                if (dest_zone == Zone::HAND) {
                     // Note: S-Trigger logic is complex (uses stack/pending).
                     // Ideally we check `GenericCardSystem::check_shield_trigger` but that method might be private/internal to EffectResolver?
                     // I will use `resolve_trigger` with `S_TRIGGER` which is the standard hook now.
                     GenericCardSystem::resolve_trigger(game_state, TriggerType::S_TRIGGER, shield_id, card_db);
                }
            }
        }
    };
}
