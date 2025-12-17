#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/systems/game_logic_system.hpp"

namespace dm::engine {

    class CastSpellHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& /*ctx*/) override {
             // See CastSpellHandler logic; mainly relies on targets.
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            using namespace dm::engine::systems;

            if (!ctx.targets || ctx.targets->empty()) return;

            for (int target_id : *ctx.targets) {
                CardInstance* card = ctx.game_state.get_card_instance(target_id);
                if (!card) continue;

                PlayerID controller = ctx.game_state.active_player_id; // Default assumption
                CardInstance* source_card = ctx.game_state.get_card_instance(ctx.source_instance_id);
                if (source_card) {
                    controller = source_card->owner;
                    if (ctx.game_state.card_owner_map.size() > (size_t)ctx.source_instance_id) {
                         controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
                    }
                }

                // Remove from current zone
                std::optional<CardInstance> removed_card = ZoneUtils::find_and_remove(ctx.game_state, target_id);
                if (!removed_card) continue;

                // Add to Stack
                ctx.game_state.stack_zone.push_back(*removed_card);

                const std::map<CardID, CardDefinition>* db_ptr = &ctx.card_db;
                std::map<CardID, CardDefinition> temp_db; // Keep alive during call

                if (ctx.action.cast_spell_side) {
                     if (ctx.card_db.count(removed_card->card_id)) {
                          const auto& def = ctx.card_db.at(removed_card->card_id);
                          if (def.spell_side) {
                               temp_db = ctx.card_db; // Copy
                               temp_db[removed_card->card_id] = *def.spell_side;
                               temp_db[removed_card->card_id].id = removed_card->card_id;
                               db_ptr = &temp_db;
                          }
                     }
                }

                // Resolve via GameLogicSystem
                // Cast Spell is essentially Play Card from Stack (free)
                Action resolve_act;
                resolve_act.type = ActionType::RESOLVE_PLAY;
                resolve_act.source_instance_id = target_id;
                resolve_act.spawn_source = SpawnSource::EFFECT_SUMMON; // Free/Effect based

                GameLogicSystem::resolve_action_oneshot(ctx.game_state, resolve_act, *db_ptr);
            }
        }
    };
}
