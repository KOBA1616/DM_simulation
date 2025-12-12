#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/effects/effect_resolver.hpp"

namespace dm::engine {

    class CastSpellHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& /*ctx*/) override {
             // See CastSpellHandler logic; mainly relies on targets.
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (!ctx.targets || ctx.targets->empty()) return;

            for (int target_id : *ctx.targets) {
                CardInstance* card = ctx.game_state.get_card_instance(target_id);
                if (!card) continue;

                // 1. Move card to stack zone (if not already there? usually from hand/grave/deck)

                PlayerID controller = ctx.game_state.active_player_id; // Default assumption
                CardInstance* source_card = ctx.game_state.get_card_instance(ctx.source_instance_id);
                if (source_card) {
                    controller = source_card->owner;
                     // Or fallback to owner map if instance is buffer/gone but map persists
                    if (ctx.game_state.card_owner_map.size() > (size_t)ctx.source_instance_id) {
                         controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
                    }
                }

                // Determine origin zone (just for robustness, we use ZoneUtils::find_and_remove)
                // Wait, find_and_remove returns the card but doesn't tell us where it came from unless we modify it.
                // But we don't strictly need to know where it came from for logic, except maybe for some triggers?
                // `resolve_play_from_stack` handles basic logic.

                // Remove from current zone
                std::optional<CardInstance> removed_card = ZoneUtils::find_and_remove(ctx.game_state, target_id);
                if (!removed_card) continue;

                // Add to Stack
                ctx.game_state.stack_zone.push_back(*removed_card);

                // Now resolve it
                // We pass cost_reduction=999 to simulate free cast (if appropriate, or 0 if paid?)
                // CAST_SPELL usually implies "Cast this spell" (often free, e.g. from S-Trigger or Effect).
                // If it requires payment, it would be PLAY_CARD action.
                // Let's assume free (cost reduction = 999) or use card cost?
                // If it's "Cast spell from graveyard" effect, it's usually free.
                // We'll use 999 for now as atomic "Cast" usually implies effect-driven cast.

                EffectResolver::resolve_play_from_stack(
                    ctx.game_state,
                    target_id,
                    999, // cost_reduction (Free)
                    SpawnSource::EFFECT_SUMMON,
                    controller,
                    ctx.card_db,
                    -1, // evo_source_id
                    0   // dest_override
                );
            }
        }
    };
}
