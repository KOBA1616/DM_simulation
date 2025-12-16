#include "legacy_json_adapter.hpp"
#include "core/types.hpp"
#include <iostream>
#include <string>

namespace dm::engine::systems {
    using namespace core;

    std::vector<Instruction> LegacyJsonAdapter::convert(const EffectDef& effect) {
        std::vector<Instruction> instructions;

        // Condition Check Wrapping
        bool has_condition = (effect.condition.type != "NONE" && !effect.condition.type.empty());

        std::vector<Instruction> inner_instructions;
        for (const auto& action : effect.actions) {
            convert_action(action, inner_instructions);
        }

        if (has_condition) {
            Instruction if_inst;
            if_inst.op = InstructionOp::IF;

            nlohmann::json cond_json;
            cond_json["type"] = effect.condition.type;
            cond_json["value"] = effect.condition.value;
            cond_json["str_val"] = effect.condition.str_val;
            cond_json["op"] = effect.condition.op;
            cond_json["stat_key"] = effect.condition.stat_key;

            if_inst.args["cond"] = cond_json;
            if_inst.then_block = inner_instructions;

            instructions.push_back(if_inst);
        } else {
            instructions = inner_instructions;
        }

        return instructions;
    }

    void LegacyJsonAdapter::convert_action(const ActionDef& action, std::vector<Instruction>& out) {
        Instruction inst;

        // 1. Check if selection is required
        bool needs_selection = false;
        if (!action.filter.zones.empty() || !action.filter.races.empty() ||
            !action.filter.civilizations.empty() ||
            action.scope == TargetScope::TARGET_SELECT ||
            action.scope == TargetScope::RANDOM ||
            action.scope == TargetScope::ALL_FILTERED) {
            needs_selection = true;
        }

        if (action.type == EffectActionType::DRAW_CARD && action.scope == TargetScope::NONE) {
            needs_selection = false;
        }

        // Output variable name for selection
        std::string target_var = "$selection_" + std::to_string(out.size()); // Unique-ish name

        if (needs_selection) {
             Instruction sel;
             sel.op = InstructionOp::SELECT;

             nlohmann::json f;
             f["zones"] = action.filter.zones;
             f["civilizations"] = action.filter.civilizations;
             f["races"] = action.filter.races;
             if (action.filter.min_cost.has_value()) f["min_cost"] = action.filter.min_cost.value();
             if (action.filter.max_cost.has_value()) f["max_cost"] = action.filter.max_cost.value();
             f["owner"] = action.filter.owner.value_or("SELF");

             sel.args["filter"] = f;
             sel.args["count"] = action.value1; // Typically value1 is count
             sel.args["out"] = target_var;
             out.push_back(sel);
        }

        // 2. Map Action Type
        switch (action.type) {
            case EffectActionType::DRAW_CARD:
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = "DECK_TOP"; // Keyword for Executor
                inst.args["to"] = "HAND";
                inst.args["count"] = action.value1;
                break;

            case EffectActionType::ADD_MANA:
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = "DECK_TOP";
                inst.args["to"] = "MANA";
                inst.args["count"] = action.value1;
                break;

            case EffectActionType::DESTROY:
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = needs_selection ? target_var : "$source";
                inst.args["to"] = "GRAVEYARD";
                break;

            case EffectActionType::SEND_TO_MANA:
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = needs_selection ? target_var : "$source";
                inst.args["to"] = "MANA";
                break;

            case EffectActionType::RETURN_TO_HAND:
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = needs_selection ? target_var : "$source";
                inst.args["to"] = "HAND";
                break;

            case EffectActionType::TAP:
                inst.op = InstructionOp::MODIFY;
                inst.args["target"] = needs_selection ? target_var : "$source";
                inst.args["type"] = "TAP";
                break;

            case EffectActionType::UNTAP:
                inst.op = InstructionOp::MODIFY;
                inst.args["target"] = needs_selection ? target_var : "$source";
                inst.args["type"] = "UNTAP";
                break;

            case EffectActionType::MODIFY_POWER:
                inst.op = InstructionOp::MODIFY;
                inst.args["target"] = needs_selection ? target_var : "$source";
                inst.args["type"] = "POWER_ADD";
                inst.args["value"] = action.value1;
                break;

            case EffectActionType::GRANT_KEYWORD:
                inst.op = InstructionOp::MODIFY;
                inst.args["target"] = needs_selection ? target_var : "$source";
                inst.args["type"] = "ADD_KEYWORD";
                inst.args["str_value"] = action.str_val; // e.g. "speed_attacker"
                break;

            case EffectActionType::MOVE_CARD:
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = needs_selection ? target_var : "$source";
                inst.args["to"] = action.destination_zone;
                break;

            case EffectActionType::DISCARD:
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = needs_selection ? target_var : "$source"; // Or selection
                inst.args["to"] = "GRAVEYARD";
                break;

            case EffectActionType::SEARCH_DECK:
                if (needs_selection) {
                    // 1. Move Selection to Hand
                    inst.op = InstructionOp::MOVE;
                    inst.args["target"] = target_var;
                    inst.args["to"] = "HAND";
                    out.push_back(inst);

                    // 2. Shuffle Deck (Implicit for Search Deck)
                    Instruction shuf;
                    shuf.op = InstructionOp::MODIFY;
                    shuf.args["type"] = "SHUFFLE";
                    shuf.args["target"] = "DECK"; // Target Zone
                    out.push_back(shuf);
                    return; // Done
                } else {
                    inst.op = InstructionOp::PRINT;
                    inst.args["msg"] = "LegacyJsonAdapter: SEARCH_DECK without explicit selection/filter logic handling.";
                }
                break;

            case EffectActionType::SHUFFLE_DECK:
                inst.op = InstructionOp::MODIFY;
                inst.args["type"] = "SHUFFLE";
                inst.args["target"] = "DECK";
                break;

            default:
                inst.op = InstructionOp::PRINT;
                inst.args["msg"] = "LegacyJsonAdapter: Unsupported action type " + std::to_string((int)action.type);
                break;
        }

        out.push_back(inst);
    }
}
