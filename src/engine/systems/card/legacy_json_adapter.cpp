#include "legacy_json_adapter.hpp"
#include "core/types.hpp"
#include <iostream>
#include <string>

namespace dm::engine::systems {
    using namespace core;

    std::vector<Instruction> LegacyJsonAdapter::convert(const EffectDef& effect) {
        std::vector<Instruction> instructions;

        // Condition Check Wrapping
        // If the effect has a condition (e.g., Mana Armed, Shield Count),
        // we wrap the actions in an IF instruction.
        // Legacy 'condition' is typically checked BEFORE resolution in the old resolver.
        // In the new pipeline, checking it inside the pipeline is robust.
        // However, we need to map ConditionDef to JSON condition structure.
        bool has_condition = (effect.condition.type != "NONE" && !effect.condition.type.empty());

        std::vector<Instruction> inner_instructions;
        for (const auto& action : effect.actions) {
            convert_action(action, inner_instructions);
        }

        if (has_condition) {
            Instruction if_inst;
            if_inst.op = InstructionOp::IF;

            // Map ConditionDef to instruction args["cond"]
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

        // Special Case: DRAW_CARD logic in legacy often implies 'Self Deck Top'.
        // Selection is implicit unless it's a "Search" effect.
        if (action.type == EffectActionType::DRAW_CARD && action.scope == TargetScope::NONE) {
            needs_selection = false;
        }

        // Output variable name for selection
        std::string target_var = "$selection_" + std::to_string(out.size()); // Unique-ish name

        if (needs_selection) {
             Instruction sel;
             sel.op = InstructionOp::SELECT;

             // Manual FilterDef to JSON mapping since we can't depend on to_json being available/correct in core
             nlohmann::json f;
             f["zones"] = action.filter.zones;
             // Helper to convert Civ enums to string if needed, but FilterDef stores Enums usually?
             // Bindings.cpp says FilterDef::civilizations is vector<Civilization>.
             // JSON serializer for Civilization enum is needed or we cast to int.
             // nlohmann::json handles enums as ints by default unless NLOHMANN_JSON_SERIALIZE_ENUM is defined.
             // Let's assume int is fine for internal pipeline.
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
        // Note: Legacy 'value1' often means 'Count' for Draw/Discard, or 'Power' for Buff.

        switch (action.type) {
            case EffectActionType::DRAW_CARD:
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = "DECK_TOP"; // Keyword for Executor
                inst.args["to"] = "HAND";
                inst.args["count"] = action.value1;
                // If selection was done (e.g. Draw X specific cards?), use var.
                // But legacy DRAW is usually top deck.
                break;

            case EffectActionType::ADD_MANA:
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = "DECK_TOP";
                inst.args["to"] = "MANA";
                inst.args["count"] = action.value1;
                break;

            case EffectActionType::DESTROY:
                inst.op = InstructionOp::MOVE;
                // If needs_selection is true, use the variable. Else implicit source?
                // Destroy usually requires a target (ActionDef::scope or Filter).
                // If atomic destroy (like "destroy self"), implicit.
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
                 // Generic Move
                inst.op = InstructionOp::MOVE;
                inst.args["target"] = needs_selection ? target_var : "$source";
                inst.args["to"] = action.destination_zone;
                break;

            case EffectActionType::DISCARD:
                inst.op = InstructionOp::MOVE;
                // Discard usually targets self hand or random.
                // If selection was required (RANDOM scope or TARGET_SELECT), use target_var.
                // If scope is NONE or SELF, it implies manual selection or random depending on context.
                // Legacy logic is complex, but mapping to MOVE(HAND -> GRAVEYARD) is the core.
                inst.args["target"] = needs_selection ? target_var : "$source"; // Or selection
                inst.args["to"] = "GRAVEYARD";
                break;

            case EffectActionType::SEARCH_DECK:
                // Search Deck involves: SELECT (implicit filter=DECK) -> MOVE(HAND) -> SHUFFLE
                // Since we handled selection above based on zones, if zones included DECK, 'target_var' holds the cards.
                // But SEARCH_DECK implies we look at the deck.
                // If 'needs_selection' didn't trigger (e.g. legacy filter was implied), we might need to force it.
                // Typically SEARCH_DECK has FilterDef with zone=DECK.
                if (needs_selection) {
                    inst.op = InstructionOp::MOVE;
                    inst.args["target"] = target_var;
                    inst.args["to"] = "HAND";
                    // Shuffle is implied. Add shuffle op?
                    // For now, assume engine handles shuffle on deck access or add separate op.
                    // We'll leave shuffle for now as it requires Deck Shuffle Command.
                } else {
                    // Fallback if no selection defined (e.g. Search any creature)
                    // We should have generated a SELECT above.
                    inst.op = InstructionOp::PRINT;
                    inst.args["msg"] = "LegacyJsonAdapter: SEARCH_DECK without explicit selection/filter logic handling.";
                }
                break;

            case EffectActionType::SHUFFLE_DECK:
                // TODO: Implement Shuffle Op
                inst.op = InstructionOp::PRINT;
                inst.args["msg"] = "LegacyJsonAdapter: SHUFFLE_DECK not yet supported in Pipeline.";
                break;

            default:
                // For unsupported actions, print warning or NOOP
                inst.op = InstructionOp::PRINT;
                inst.args["msg"] = "LegacyJsonAdapter: Unsupported action type " + std::to_string((int)action.type);
                break;
        }

        out.push_back(inst);
    }
}
