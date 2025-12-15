#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include <iostream>

namespace dm::engine {

    class GrantKeywordHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::engine::game_command;

            // GRANT_KEYWORD uses ADD_PASSIVE_EFFECT in MutateCommand.
            // str_value will be the keyword string.
            // We pass it directly. MutateCommand logic for ADD_PASSIVE_EFFECT defaults to KEYWORD_GRANT if str_value is not POWER/LOCK_SPELL.

            int duration = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;
            int source = ctx.source_instance_id;
            dm::core::PlayerID controller = ctx.game_state.card_owner_map[source];

            auto cmd = std::make_shared<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT, 0, ctx.action.str_val);
            cmd->set_modifier_data(ctx.action.filter, duration, controller, source);

            cmd->execute(ctx.game_state);
            ctx.game_state.command_history.push_back(cmd);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            // As noted in previous analysis, target-specific grant keyword usually implies a filter derived from selection,
            // OR we need specific ID support in PassiveEffect.
            // For now, we assume the filter in `ctx.action` is sufficient or we rely on the `resolve` implementation.
            // If targets were selected, we might want to restrict the filter.
            // But `ModifierHandler` also just passed `ctx.action.filter`.
            // If `RESOLVE_EFFECT` (which calls this) set up the filter based on targets, we are good.
            // If `RESOLVE_EFFECT` passed explicit targets but the action filter is generic...
            // `GenericCardSystem::select_targets` creates a PendingEffect.
            // When resolved, `resolve_effect_with_targets` is called.
            // But `GrantKeywordHandler` adds a PASSIVE EFFECT.
            // A PassiveEffect needs a FilterDef.
            // It does not accept a list of IDs.
            // So we strictly rely on `ctx.action.filter`.
            // If the user selected targets, that selection is effectively ignored unless we dynamically construct a filter "ID in [targets]".
            // This is a known limitation of current PassiveEffect system vs Target Selection.
            // However, most "Grant Keyword" effects are "All your creatures get X" (Filter based).
            // "Target creature gets X" is usually implemented as a `MUTATE` (Power/Flags) if it's permanent?
            // If it is "until end of turn", it requires a PassiveEffect targeting that specific ID.
            // TODO: Future task -> Support ID-based PassiveEffects.

            resolve(ctx);
        }
    };
}
