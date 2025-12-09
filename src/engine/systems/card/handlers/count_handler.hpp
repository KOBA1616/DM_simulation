#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include <set>
#include <string>
#include <algorithm>

namespace dm::engine {

    class CountHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
            Player& controller = ctx.game_state.players[controller_id];

             if (ctx.action.type == EffectActionType::COUNT_CARDS) {
                int count = 0;
                const auto& f = ctx.action.filter;

                auto check_zone = [&](const std::vector<CardInstance>& zone, int owner_id) {
                     for (const auto& card : zone) {
                         const CardData* cd = CardRegistry::get_card_data(card.card_id);
                         if (!cd) continue;

                         if (!f.types.empty()) {
                             bool match = false;
                             for(auto& t : f.types) if(t == cd->type) match = true;
                             if(!match) continue;
                         }
                         if (!f.civilizations.empty()) {
                             bool match = false;
                             for(auto& fc : f.civilizations) {
                                 for(auto& cc : cd->civilizations) {
                                     if(fc == cc) {
                                         match = true;
                                         break;
                                     }
                                 }
                                 if(match) break;
                             }
                             if(!match) continue;
                         }
                         if (!f.races.empty()) {
                             bool match = false;
                             for(auto& r : f.races) {
                                 for(auto& cr : cd->races) if(r == cr) match = true;
                             }
                             if(!match) continue;
                         }
                         if (f.owner.has_value()) {
                             if (f.owner == "SELF" && owner_id != controller_id) continue;
                             if (f.owner == "OPPONENT" && owner_id == controller_id) continue;
                         }

                         count++;
                     }
                };

                for (const auto& z : f.zones) {
                    if (z == "BATTLE_ZONE") {
                        check_zone(ctx.game_state.players[0].battle_zone, 0);
                        check_zone(ctx.game_state.players[1].battle_zone, 1);
                    } else if (z == "GRAVEYARD") {
                        check_zone(ctx.game_state.players[0].graveyard, 0);
                        check_zone(ctx.game_state.players[1].graveyard, 1);
                    } else if (z == "MANA_ZONE") {
                         check_zone(ctx.game_state.players[0].mana_zone, 0);
                         check_zone(ctx.game_state.players[1].mana_zone, 1);
                    } else if (z == "HAND") {
                         check_zone(ctx.game_state.players[0].hand, 0);
                         check_zone(ctx.game_state.players[1].hand, 1);
                    } else if (z == "SHIELD_ZONE") {
                         check_zone(ctx.game_state.players[0].shield_zone, 0);
                         check_zone(ctx.game_state.players[1].shield_zone, 1);
                    }
                }

                if (!ctx.action.output_value_key.empty()) {
                    ctx.execution_vars[ctx.action.output_value_key] = count;
                }
            } else if (ctx.action.type == EffectActionType::GET_GAME_STAT) {
                int result = 0;
                if (ctx.action.str_val == "MANA_CIVILIZATION_COUNT") {
                    std::set<std::string> civs;
                    for (const auto& c : controller.mana_zone) {
                        const CardData* cd = CardRegistry::get_card_data(c.card_id);
                        if (cd) {
                             for (const auto& civ_str : cd->civilizations) {
                                 civs.insert(civ_str);
                             }
                        }
                    }
                    result = (int)civs.size();
                } else if (ctx.action.str_val == "SHIELD_COUNT") {
                    result = (int)controller.shield_zone.size();
                } else if (ctx.action.str_val == "HAND_COUNT") {
                    result = (int)controller.hand.size();
                } else if (ctx.action.str_val == "CARDS_DRAWN_THIS_TURN") {
                    result = ctx.game_state.turn_stats.cards_drawn_this_turn;
                }

                if (!ctx.action.output_value_key.empty()) {
                    ctx.execution_vars[ctx.action.output_value_key] = result;
                }
            }
        }
    };
}
