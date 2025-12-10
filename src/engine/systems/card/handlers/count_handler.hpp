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
                         if (!ctx.card_db.count(card.card_id)) continue;
                         const auto& cd = ctx.card_db.at(card.card_id);

                         // Check types (CardDefinition uses CardType enum, Filter uses string)
                         if (!f.types.empty()) {
                             bool match = false;
                             // This part requires converting cd.type (enum) to string to match f.types
                             // or f.types to enum.
                             // Assuming for now simple "CREATURE"/"SPELL" match via helper or if removed.
                             // Implementing basic check:
                             std::string type_str = "CREATURE";
                             if (cd.type == CardType::SPELL) type_str = "SPELL";
                             for(auto& t : f.types) if(t == type_str) match = true;
                             if(!match) continue;
                         }
                         if (!f.civilizations.empty()) {
                             bool match = false;
                             for(auto& fc : f.civilizations) {
                                 // cd.civilizations is vector<Civilization>
                                 for(auto& cc : cd.civilizations) {
                                     if (fc == "LIGHT" && cc == Civilization::LIGHT) match = true;
                                     if (fc == "WATER" && cc == Civilization::WATER) match = true;
                                     if (fc == "DARKNESS" && cc == Civilization::DARKNESS) match = true;
                                     if (fc == "FIRE" && cc == Civilization::FIRE) match = true;
                                     if (fc == "NATURE" && cc == Civilization::NATURE) match = true;
                                     if (fc == "ZERO" && cc == Civilization::ZERO) match = true;
                                     if(match) break;
                                 }
                                 if(match) break;
                             }
                             if(!match) continue;
                         }
                         if (!f.races.empty()) {
                             bool match = false;
                             for(auto& r : f.races) {
                                 for(auto& cr : cd.races) if(r == cr) match = true;
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
                        if (ctx.card_db.count(c.card_id)) {
                             const auto& cd = ctx.card_db.at(c.card_id);
                             for (const auto& civ : cd.civilizations) {
                                 if (civ == Civilization::LIGHT) civs.insert("LIGHT");
                                 if (civ == Civilization::WATER) civs.insert("WATER");
                                 if (civ == Civilization::DARKNESS) civs.insert("DARKNESS");
                                 if (civ == Civilization::FIRE) civs.insert("FIRE");
                                 if (civ == Civilization::NATURE) civs.insert("NATURE");
                                 if (civ == Civilization::ZERO) civs.insert("ZERO");
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
