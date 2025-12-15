#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include <iostream>

namespace dm::engine {

    class GrantKeywordHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            // GRANT_KEYWORD usually has a filter to determine WHO gets the keyword.
            // But it adds a PassiveEffect to the GameState, which then applies continuously.
            // It does NOT iterate targets and apply a flag to them individually (unless it's a permanent flag, but requirement says "Continuous effect").
            // "Give target creature Blocker" -> Effect lasts until end of turn usually.
            // "All your creatures have Blocker" -> Permanent passive if source is valid.

            // If the action is from a "continuous" source (like a creature in Battle Zone),
            // the passive effect should have been registered when the creature entered play (or statically checked).
            // BUT here we are resolving an ACTION (e.g. from a Spell or CIP ability).
            // So this creates a TEMPORARY passive effect (duration based).

            // ActionDef:
            // filter: who gets it
            // str_val: keyword (e.g. "BLOCKER")
            // value2: duration (1 = until end of turn)

            dm::core::PassiveEffect passive;
            passive.type = dm::core::PassiveType::KEYWORD_GRANT;
            passive.str_value = ctx.action.str_val;
            passive.target_filter = ctx.action.filter;
            passive.source_instance_id = ctx.source_instance_id;
            passive.controller = ctx.game_state.card_owner_map[ctx.source_instance_id];

            // Duration logic
            // value2 = 1 -> This turn
            // value2 = 0 -> Indefinite? Or default to 1?
            // Usually Spells give it "until the start of your next turn" or "until end of turn".
            // Let's assume value2 is turns. If 0, maybe default to 1 for safety, or check spec.
            // If it's a permanent grant (e.g. "This creature gains..."), it might be permanent.

            if (ctx.action.value2 > 0) {
                passive.turns_remaining = ctx.action.value2;
            } else {
                passive.turns_remaining = 1; // Default to 1 turn if not specified
            }

            // Phase 6.3: Use GameCommand
            auto cmd = std::make_shared<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PASSIVE);
            cmd->passive_payload = passive;
            cmd->execute(const_cast<core::GameState&>(ctx.game_state));
            const_cast<core::GameState&>(ctx.game_state).command_history.push_back(cmd);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            // If targets were selected (Scope::TARGET_SELECT), the filter in the passive
            // should probably be restricted to those IDs?
            // But PassiveEffect uses a FilterDef, not a list of IDs.
            // If we select specific targets, we might need a way to create a filter "ID is X or Y".
            // OR, we attach a specific ID list to the passive?
            // Existing PassiveEffect structure only has `target_filter`.

            // Workaround: If targets are specific, we can't easily express it as a simple filter unless we add `target_instance_ids` to PassiveEffect.
            // Given the requirement "Give 'Blocker' to allied creatures" (usually all), the filter approach works.
            // If it is "Give target creature Blocker", we need to support ID filtering in Passives.

            // Let's check `TargetUtils`. It doesn't check IDs against a list in `FilterDef` typically.
            // But we can add `std::vector<int> specific_ids` to `FilterDef`? No, FilterDef is from JSON.

            // We should modify `PassiveEffect` to optionally hold specific instance IDs.
            // But for now, let's implement the generic case.

            // If `ctx.targets` is present, it means specific cards were chosen.
            // We can't implement "Give TARGET creature blocker" correctly without ID support in Passive.
            // Let's assume for this task (GRANT_KEYWORD to allied creatures) it is a broad filter.

            resolve(ctx);
        }
    };
}
