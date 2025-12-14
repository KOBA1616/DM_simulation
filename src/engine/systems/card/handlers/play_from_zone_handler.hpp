#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/effects/effect_resolver.hpp"
#include "engine/systems/mana/mana_system.hpp"

namespace dm::engine {

    class PlayFromZoneHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& /*ctx*/) override {
             // Usually requires targets
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            const auto& action = ctx.action;
            GameState& game_state = ctx.game_state;

            if (!ctx.targets || ctx.targets->empty()) return;

            // Flags from ActionDef
            bool pay_cost = action.pay_cost;
            bool as_summon = action.as_summon;

            // Determine Controller (usually Active Player for "Play" unless specified)
            PlayerID controller_id = game_state.active_player_id;
            Player& player = game_state.players[controller_id];

            for (int target_id : *ctx.targets) {
                // 1. Remove from source zone
                std::optional<CardInstance> removed_opt = ZoneUtils::find_and_remove(game_state, target_id);
                if (!removed_opt) continue;

                CardInstance card = *removed_opt;

                // 2. Cost Payment Logic
                int cost_reduction = 0;
                bool can_play = true;

                if (pay_cost) {
                    if (ctx.card_db.count(card.card_id)) {
                         const auto& def = ctx.card_db.at(card.card_id);
                         // Check if affordable
                         if (ManaSystem::can_pay_cost(game_state, player, def, ctx.card_db)) {
                              // Perform Payment
                              ManaSystem::auto_tap_mana(game_state, player, def, ctx.card_db);
                              card.is_tapped = true; // Mark as paid/tapped
                         } else {
                              can_play = false;
                         }
                    }
                } else {
                    // Free play
                    cost_reduction = 999;
                }

                if (!can_play) {
                    // Revert to Graveyard
                    player.graveyard.push_back(card);
                    continue;
                }

                // 3. Move to Stack for Resolution
                game_state.stack_zone.push_back(card);

                // 4. Resolve Play
                SpawnSource spawn_source = as_summon ? SpawnSource::EFFECT_SUMMON : SpawnSource::EFFECT_PUT;

                EffectResolver::resolve_play_from_stack(
                    game_state,
                    card.instance_id,
                    cost_reduction,
                    spawn_source,
                    controller_id,
                    ctx.card_db
                );
            }
        }
    };
}
