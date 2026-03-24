#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace dm::core {

    enum class InstructionOp {
        NOOP,
        // Logic
        IF,
        LOOP,       // FOREACH
        REPEAT,     // Loop N times
        // Query
        SELECT,
        GET_STAT,   // Get Game State (Hand count, etc.)
        // Action
        MOVE,       // Zone change
        MODIFY,     // Power/Tap/Shield
        GAME_ACTION,// High-level Game Logic (Play, Attack, etc.)
        // High-level specific (mapped to GAME_ACTION internally or separate)
        PLAY,
        ATTACK,
        BLOCK,
        // Calc
        COUNT,
        MATH,
        // Flow
        CALL,
        RETURN,
        WAIT_INPUT, // Added for interactive pipeline
        // Debug
        PRINT
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(InstructionOp, {
        {InstructionOp::NOOP, "NOOP"},
        {InstructionOp::IF, "IF"},
        {InstructionOp::LOOP, "LOOP"},
        {InstructionOp::REPEAT, "REPEAT"},
        {InstructionOp::SELECT, "SELECT"},
        {InstructionOp::GET_STAT, "GET_STAT"},
        {InstructionOp::MOVE, "MOVE"},
        {InstructionOp::MODIFY, "MODIFY"},
        {InstructionOp::GAME_ACTION, "GAME_ACTION"},
        {InstructionOp::PLAY, "PLAY"},
        {InstructionOp::ATTACK, "ATTACK"},
        {InstructionOp::BLOCK, "BLOCK"},
        {InstructionOp::COUNT, "COUNT"},
        {InstructionOp::MATH, "MATH"},
        {InstructionOp::CALL, "CALL"},
        {InstructionOp::RETURN, "RETURN"},
        {InstructionOp::WAIT_INPUT, "WAIT_INPUT"},
        {InstructionOp::PRINT, "PRINT"}
    })

    struct Instruction {
        InstructionOp op = InstructionOp::NOOP;

        // Arguments stored as JSON for flexibility
        // Example args: "filter", "count", "out", "cond", "target", "to", "value"
        nlohmann::json args;

        // Recursive blocks for control flow
        std::vector<Instruction> then_block;
        std::vector<Instruction> else_block;

        // Constructors
        Instruction() = default;
        Instruction(InstructionOp o) : op(o) {}
        Instruction(InstructionOp o, nlohmann::json a) : op(o), args(a) {}
    };

    // Forward declaration for recursion
    inline void from_json(const nlohmann::json& j, Instruction& i);

    inline void to_json(nlohmann::json& j, const Instruction& i) {
        j = i.args; // Start with args
        j["op"] = i.op;
        if (!i.then_block.empty()) j["then"] = i.then_block;
        if (!i.else_block.empty()) j["else"] = i.else_block;
    }

    inline void from_json(const nlohmann::json& j, Instruction& i) {
        // Defensive parsing: allow `j` to be null or non-object without throwing
        if (!j.is_null() && j.contains("op")) {
            j.at("op").get_to(i.op);
        } else {
            i.op = InstructionOp::NOOP;
        }

        // Ensure args is always an object (not null) for downstream code
        if (j.is_null() || j.is_array()) {
            i.args = nlohmann::json::object();
        } else {
            i.args = j;
        }

        // Remove reserved keys from args if present
        if (i.args.is_object()) {
            i.args.erase("op");
            i.args.erase("then");
            i.args.erase("else");
        }

        // Parse blocks safely
        if (!j.is_null() && j.contains("then")) {
             j.at("then").get_to(i.then_block);
        }
        if (!j.is_null() && j.contains("else")) {
             j.at("else").get_to(i.else_block);
        }
    }

}
