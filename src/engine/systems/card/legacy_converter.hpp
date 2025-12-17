#pragma once
#include "core/card_json_types.hpp"
#include <optional>

namespace dm::engine::systems {

    class LegacyConverter {
    public:
        static std::optional<dm::core::CommandDef> convert(const dm::core::ActionDef& action) {
            dm::core::CommandDef cmd;

            // Map Condition (Direct copy since types are compatible/same struct)
            if (action.condition.has_value()) {
                cmd.condition = action.condition;
            }

            switch (action.type) {
                case dm::core::EffectActionType::DRAW_CARD:
                    cmd.type = dm::core::CommandType::DRAW_CARD;
                    cmd.amount = (action.value1 > 0) ? action.value1 : 1; // Default to 1 if 0
                    return cmd;

                case dm::core::EffectActionType::ADD_MANA:
                    cmd.type = dm::core::CommandType::MANA_CHARGE;
                    cmd.amount = (action.value1 > 0) ? action.value1 : 1;
                    return cmd;

                case dm::core::EffectActionType::DESTROY:
                    cmd.type = dm::core::CommandType::DESTROY;
                    cmd.target_group = action.scope;
                    cmd.target_filter = action.filter;
                    // Legacy DESTROY usually implies checking the filter zones.
                    // If filter.zones is empty, default to BATTLE_ZONE for Destroy.
                    if (cmd.target_filter.zones.empty()) {
                         cmd.target_filter.zones.push_back("BATTLE_ZONE");
                    }
                    return cmd;

                case dm::core::EffectActionType::TAP:
                    cmd.type = dm::core::CommandType::TAP;
                    cmd.target_group = action.scope;
                    cmd.target_filter = action.filter;
                    if (cmd.target_filter.zones.empty()) {
                        cmd.target_filter.zones.push_back("BATTLE_ZONE");
                    }
                    return cmd;

                case dm::core::EffectActionType::UNTAP:
                    cmd.type = dm::core::CommandType::UNTAP;
                    cmd.target_group = action.scope;
                    cmd.target_filter = action.filter;
                     if (cmd.target_filter.zones.empty()) {
                        cmd.target_filter.zones.push_back("BATTLE_ZONE");
                        cmd.target_filter.zones.push_back("MANA_ZONE"); // Untap often targets mana too, but filter usually specifies
                    }
                    return cmd;

                case dm::core::EffectActionType::RETURN_TO_HAND:
                     cmd.type = dm::core::CommandType::RETURN_TO_HAND;
                     cmd.target_group = action.scope;
                     cmd.target_filter = action.filter;
                     if (cmd.target_filter.zones.empty()) {
                         cmd.target_filter.zones.push_back("BATTLE_ZONE");
                         cmd.target_filter.zones.push_back("MANA_ZONE");
                     }
                     return cmd;

                default:
                    return std::nullopt;
            }
        }
    };

}
