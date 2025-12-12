#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/effects/effect_resolver.hpp"

namespace dm::engine {

    class PutCreatureHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& /*ctx*/) override {
             // See CastSpellHandler logic; mainly relies on targets.
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (!ctx.targets || ctx.targets->empty()) return;

            for (int target_id : *ctx.targets) {

                // Use find_and_remove instead of manual scan
                std::optional<CardInstance> removed_opt = ZoneUtils::find_and_remove(ctx.game_state, target_id);
                if (!removed_opt) continue;

                CardInstance card = *removed_opt;

                // Determine controller (usually the one invoking, or specified)
                PlayerID controller = ctx.game_state.active_player_id;
                CardInstance* source = ctx.game_state.get_card_instance(ctx.source_instance_id);
                if (source) {
                    controller = source->owner;
                    if (ctx.game_state.card_owner_map.size() > (size_t)ctx.source_instance_id) {
                         controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
                    }
                }

                // Set Battle Zone State
                card.summoning_sickness = true;
                // Check SA/MachFighter later? Triggers handle it?
                // Actually SA is static ability.

                card.turn_played = ctx.game_state.turn_number;

                // Add to Battle Zone
                ctx.game_state.players[controller].battle_zone.push_back(card);
                if (ctx.game_state.card_owner_map.size() <= (size_t)target_id) {
                    ctx.game_state.card_owner_map.resize(target_id + 1, 255);
                }
                ctx.game_state.card_owner_map[target_id] = controller;

                // Trigger ON_PLAY / ON_OTHER_ENTER
                GenericCardSystem::resolve_trigger(ctx.game_state, TriggerType::ON_PLAY, target_id, ctx.card_db);
                GenericCardSystem::resolve_trigger(ctx.game_state, TriggerType::ON_OTHER_ENTER, target_id, ctx.card_db);
            }
        }
    };
}
