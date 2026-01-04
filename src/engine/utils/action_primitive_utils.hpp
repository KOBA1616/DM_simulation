#pragma once
#include "core/instruction.hpp"
#include <vector>
#include <string>

namespace dm::engine::utils {

    class ActionPrimitiveUtils {
    public:
        // Generates a MOVE instruction to the Mana Zone
        static dm::core::Instruction create_mana_charge_instruction(const nlohmann::json& target, int count = 1, bool to_bottom = false) {
            dm::core::Instruction move(dm::core::InstructionOp::MOVE);
            move.args["to"] = "MANA";
            move.args["target"] = target;
            if (count > 1) move.args["count"] = count;
            if (to_bottom) move.args["to_bottom"] = true;
            return move;
        }

        // Overload for integer target ID
        static dm::core::Instruction create_mana_charge_instruction(int target_id) {
            return create_mana_charge_instruction(nlohmann::json(target_id));
        }
    };
}
