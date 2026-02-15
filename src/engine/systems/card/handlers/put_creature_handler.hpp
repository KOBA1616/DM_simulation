#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "core/card_def.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
// #include "engine/effects/effect_resolver.hpp" // Removed
#include "engine/systems/effects/trigger_system.hpp"

namespace dm::engine {

    class PutCreatureHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& /*ctx*/) override {
             // See CastSpellHandler logic; mainly relies on targets.
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (!ctx.targets || ctx.targets->empty()) return;

            // Determine origin string from action definition
            std::string origin_str = ctx.action.source_zone;
            // Normalize MANA vs MANA_ZONE
            if (origin_str == "MANA") origin_str = "MANA_ZONE";
            if (origin_str == "SHIELD") origin_str = "SHIELD_ZONE";
            if (origin_str == "BATTLE") origin_str = "BATTLE_ZONE";

            for (int target_id : *ctx.targets) {

                // Pre-check for Prohibitions (CANNOT_SUMMON)
                const CardInstance* existing_card = ctx.game_state.get_card_instance(target_id);
                if (existing_card && ctx.card_db.count(existing_card->card_id)) {
                    const auto& def = ctx.card_db.at(existing_card->card_id);

                    bool prohibited = false;
                    for (const auto& eff : ctx.game_state.passive_effects) {
                        if (eff.type == PassiveType::CANNOT_SUMMON && (def.type == CardType::CREATURE || def.type == CardType::EVOLUTION_CREATURE)) {
                            bool origin_match = true;
                            if (!eff.target_filter.zones.empty()) {
                                origin_match = false;
                                if (!origin_str.empty()) {
                                    for (const auto& z : eff.target_filter.zones) {
                                        if (z == origin_str) { origin_match = true; break; }
                                    }
                                }
                            }

                            if (!origin_match) continue;

                            FilterDef check_filter = eff.target_filter;
                            check_filter.zones.clear();

                            if (dm::engine::utils::TargetUtils::is_valid_target(*existing_card, def, check_filter, ctx.game_state, eff.controller, existing_card->owner, true)) {
                                prohibited = true;
                                break;
                            }
                        }
                    }
                    if (prohibited) continue; // Skip putting this creature into play
                }

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
                         controller = ctx.game_state.get_card_owner(ctx.source_instance_id);
                    }
                }

                // Set Battle Zone State
                // If the card definition grants speed attacker, do NOT set summoning sickness
                bool has_speed = false;
                if (ctx.card_db.count(card.card_id)) {
                    const auto& cdef = ctx.card_db.at(card.card_id);
                    has_speed = cdef.keywords.speed_attacker;
                }
                card.summoning_sickness = !has_speed;
                // Check SA/MachFighter later? Triggers handle it?
                // Actually SA is static ability.

                card.turn_played = ctx.game_state.turn_number;

                // Add to Battle Zone
                ctx.game_state.players[controller].battle_zone.push_back(card);
                if (ctx.game_state.card_owner_map.size() <= (size_t)target_id) {
                    ctx.game_state.card_owner_map.resize(target_id + 1, 255);
                }
                ctx.game_state.set_card_owner(target_id, controller);

                // Trigger ON_PLAY / ON_OTHER_ENTER
                systems::TriggerSystem::instance().resolve_trigger(ctx.game_state, TriggerType::ON_PLAY, target_id, ctx.card_db);
                systems::TriggerSystem::instance().resolve_trigger(ctx.game_state, TriggerType::ON_OTHER_ENTER, target_id, ctx.card_db);
            }
        }
    };
}
