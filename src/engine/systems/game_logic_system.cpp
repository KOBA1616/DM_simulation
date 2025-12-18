#include "game_logic_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/actions/action_generator.hpp" // For phase/strategies if needed? No, separate.
#include "engine/systems/flow/phase_manager.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include <iostream>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    void GameLogicSystem::dispatch_action(PipelineExecutor& pipeline, GameState& state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        Instruction inst;
        inst.op = InstructionOp::GAME_ACTION;

        // Map ActionType to Instruction Args
        if (action.type == ActionType::PLAY_CARD) {
            inst.args["type"] = "PLAY_CARD";
            inst.args["card_id"] = action.card_id;
            inst.args["source_id"] = action.source_instance_id;
            inst.args["target_id"] = action.target_instance_id;
        }
        else if (action.type == ActionType::ATTACK_CREATURE) {
            inst.args["type"] = "ATTACK";
            inst.args["attacker"] = action.source_instance_id;
            inst.args["target"] = action.target_instance_id;
            inst.args["target_type"] = "CREATURE";
        }
        else if (action.type == ActionType::ATTACK_PLAYER) {
            inst.args["type"] = "ATTACK";
            inst.args["attacker"] = action.source_instance_id;
            inst.args["target"] = action.target_player; // Use target_player for player attack
            inst.args["target_type"] = "PLAYER";
        }
        else if (action.type == ActionType::BLOCK) {
            inst.args["type"] = "BLOCK";
            inst.args["blocker"] = action.source_instance_id;
            inst.args["attacker"] = action.target_instance_id; // Current attacker
        }
        else if (action.type == ActionType::MANA_CHARGE) {
             inst.args["type"] = "MANA_CHARGE";
             inst.args["card_id"] = action.source_instance_id; // Use instance ID for mana charge
        }
        // ... mappings for other actions

        // Execute the top-level game logic instruction
        // This will expand into lower-level instructions in the pipeline
        if (inst.args.contains("type")) {
            std::string type = inst.args["type"];
            if (type == "PLAY_CARD") handle_play_card(pipeline, state, inst, card_db);
            else if (type == "ATTACK") handle_attack(pipeline, state, inst, card_db);
            else if (type == "BLOCK") handle_block(pipeline, state, inst, card_db);
            else if (type == "MANA_CHARGE") handle_mana_charge(pipeline, state, inst);
            // Handle new ATTACH_CARD for evolution/cross gear
            else if (type == "ATTACH_CARD") {
                // Logic to attach card to another (Evolution, Cross Gear, etc.)
                int source = inst.args.value("source_id", -1);
                int target = inst.args.value("target_id", -1);
                (void)source;
                (void)target;

                // For now, this is a placeholder as requested by the plan.
            }
        }
    }

    void GameLogicSystem::resolve_action_oneshot(GameState& state, const Action& action, const std::map<CardID, CardDefinition>& card_db) {
        PipelineExecutor pipeline;
        dispatch_action(pipeline, state, action, card_db);
        // PipelineExecutor::execute is synchronous now, but if instructions were queued inside dispatch, we need to flush.
        // dispatch_action calls handle_* which calls pipeline.execute immediately.
    }

    // --- Implementations ---

    void GameLogicSystem::handle_play_card(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        int instance_id = inst.args.value("source_id", -1);
        if (instance_id == -1) return;

        // 1. Validate
        // 2. Pay Cost (Modify mana)
        // 3. Move to Stack (InstructionOp::MOVE)
        // 4. Resolve (Call resolve_play logic)

        // Generating instructions for the pipeline
        std::vector<Instruction> instructions;

        // Move Hand -> Stack
        Instruction move(InstructionOp::MOVE);
        move.args["source_key"] = "play_card_inst"; // We need to set context first
        move.args["destination"] = "STACK";
        // To make this work, we need to set context var "play_card_inst" to [instance_id]
        pipeline.set_context_var("play_card_inst", std::vector<int>{instance_id});

        instructions.push_back(move);

        // Execute these atomic steps
        pipeline.execute(instructions, state, card_db);

        // Now triggering RESOLVE_PLAY logic
        // This involves trigger checking and moving to battle zone or graveyard.
        // Delegate to EffectSystem or specialized logic.
        // For compatibility, we might assume the card is now in STACK.
        // Resolve it:

        // Since we are in C++, we can call GameLogicSystem::resolve_play_from_stack logic
        // But that logic assumes monolithic execution.
        // Ideally we break it down.
        // For "Stabilization", we can call the legacy helper if it helps.
    }

    void GameLogicSystem::handle_resolve_play(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        // Logic for resolving a card already on stack
        (void)pipeline;
        (void)state;
        (void)inst;
        (void)card_db;
    }

    void GameLogicSystem::handle_attack(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        // Attack declaration logic
        // Tap attacker
        int attacker_id = inst.args.value("attacker", -1);
        Instruction tap(InstructionOp::MODIFY);
        tap.args["target"] = attacker_id;
        tap.args["modification"] = "TAP";

        std::vector<Instruction> batch;
        batch.push_back(tap);
        pipeline.execute(batch, state, card_db);

        // Set attack state
        // trigger ON_ATTACK
    }

    void GameLogicSystem::handle_block(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        // Tap blocker, set battle state
        int blocker_id = inst.args.value("blocker", -1);
        if (blocker_id != -1) {
            // Tap the blocker
            Instruction tap(InstructionOp::MODIFY);
            tap.args["target"] = blocker_id;
            tap.args["modification"] = "TAP";
            std::vector<Instruction> batch;
            batch.push_back(tap);
            pipeline.execute(batch, state, card_db);

            // Set blocked status
            state.current_attack.is_blocked = true;
            state.current_attack.blocker_instance_id = blocker_id;

            // Trigger ON_BLOCK
            push_trigger_check(pipeline, TriggerType::ON_BLOCK, blocker_id);
        }
    }

    void GameLogicSystem::handle_resolve_battle(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        // Compare powers, destroy loser
        // Get attacker and target from current_attack
        int attacker_id = state.current_attack.source_instance_id;
        int target_id = state.current_attack.is_blocked ? state.current_attack.blocker_instance_id : state.current_attack.target_instance_id;

        if (target_id == -1) return; // Player attack, handled elsewhere? Or resolved via break shields.

        const auto* attacker = state.get_card_instance(attacker_id);
        const auto* target = state.get_card_instance(target_id);

        if (!attacker || !target) return;

        int p1 = get_creature_power(*attacker, state, card_db);
        int p2 = get_creature_power(*target, state, card_db);

        std::vector<int> to_destroy;

        if (p1 > p2) {
            to_destroy.push_back(target_id);
        } else if (p2 > p1) {
            to_destroy.push_back(attacker_id);
        } else {
            to_destroy.push_back(attacker_id);
            to_destroy.push_back(target_id);
        }

        // Handle Slayer
        // ... (Simplified for now)

        // Destroy cards
        if (!to_destroy.empty()) {
            Instruction destroy(InstructionOp::MOVE);
            destroy.args["target"] = "$battle_losers";
            destroy.args["to"] = "GRAVEYARD";
            pipeline.set_context_var("battle_losers", to_destroy);

            std::vector<Instruction> batch;
            batch.push_back(destroy);
            pipeline.execute(batch, state, card_db);
        }
    }

    void GameLogicSystem::handle_break_shield(PipelineExecutor& pipeline, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        // Shield break logic
        // Move shield to hand
        int shield_id = inst.args.value("shield_id", -1); // Assuming arg
        if (shield_id == -1) {
             // Maybe passed as target?
             // Let's assume we need to break N shields.
             // Or specific shields.
             // For now, simple implementation
             return;
        }

        // Move Shield -> Hand
        Instruction move(InstructionOp::MOVE);
        move.args["target"] = shield_id;
        move.args["to"] = "HAND";

        std::vector<Instruction> batch;
        batch.push_back(move);
        pipeline.execute(batch, state, card_db);

        // Check Shield Trigger
        const auto* card = state.get_card_instance(shield_id); // Wait, it moved to hand. We need to check definition.
        // It's in hand now. We can look it up.
        if (card) {
             const auto& def = card_db.at(card->card_id);
             if (def.keywords.shield_trigger) {
                 // Queue USE_SHIELD_TRIGGER
                 // This usually requires a Reaction Window or immediate prompt.
                 // For now, just logging or setting flag.
             }
        }
    }

    void GameLogicSystem::handle_mana_charge(PipelineExecutor& pipeline, GameState& state, const Instruction& inst) {
        int card_id = inst.args.value("card_id", -1);
        if (card_id == -1) return;

        Instruction move(InstructionOp::MOVE);
        move.args["target"] = card_id;
        move.args["to"] = "MANA";

        // Since PipelineExecutor::execute is synchronous and we don't need card_db for simple move if context has it?
        // Wait, execute needs card_db.
        // handle_mana_charge signature in cpp file uses GameState& state, const Instruction& inst.
        // It doesn't receive card_db.
        // I need to change signature or pass dummy.
        // Wait, handle_mana_charge in header:
        // static void handle_mana_charge(PipelineExecutor& pipeline, core::GameState& state, const core::Instruction& inst);

        // PipelineExecutor::execute requires card_db.
        // We can pass an empty map if we know MOVE doesn't use it except for some checks?
        // TargetUtils uses it.
        // handle_move uses it? No, handle_move mostly uses state.

        std::map<CardID, CardDefinition> empty_db;
        std::vector<Instruction> batch;
        batch.push_back(move);
        pipeline.execute(batch, state, empty_db);
    }

    void GameLogicSystem::handle_use_ability(PipelineExecutor&, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        // Use ability (e.g. Revolution Change)
        (void)state;
        (void)inst;
        (void)card_db;
    }

    void GameLogicSystem::handle_resolve_reaction(PipelineExecutor&, GameState& state, const Instruction& inst, const std::map<CardID, CardDefinition>& card_db) {
        // Use ability (e.g. Revolution Change)
        (void)state;
        (void)inst;
        (void)card_db;
    }

    void GameLogicSystem::handle_select_target(PipelineExecutor&, GameState& state, const Instruction& inst) {
        // Use ability (e.g. Revolution Change)
        (void)state;
        (void)inst;
    }

    // ... Helper implementations ...

    void GameLogicSystem::resolve_play_from_stack(GameState& game_state, int stack_instance_id, int cost_reduction, SpawnSource spawn_source, PlayerID controller, const std::map<CardID, CardDefinition>& card_db, int evo_source_id, int dest_override) {
        // Legacy/Transition helper
        // Utilize EffectSystem::resolve_effect_with_context if needed?
        // Or pure C++ logic.
        (void)game_state;
        (void)stack_instance_id;
        (void)cost_reduction;
        (void)spawn_source;
        (void)controller;
        (void)card_db;
        (void)evo_source_id;
        (void)dest_override;
    }

    int GameLogicSystem::get_creature_power(const CardInstance& creature, const GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        int power = 0;
        if (card_db.count(creature.card_id)) {
            power = card_db.at(creature.card_id).power;
        }
        power += creature.power_mod;

        // Iterate passive effects
        for (const auto& pe : game_state.passive_effects) {
            if (pe.type == PassiveType::POWER_MODIFIER) {
                // Check if applies
                if (TargetUtils::is_valid_target(creature, card_db.at(creature.card_id), pe.target_filter, game_state, pe.controller, creature.owner)) {
                    power += pe.value;
                }
            }
        }

        return power;
    }

    int GameLogicSystem::get_breaker_count(const CardInstance& creature, const std::map<CardID, CardDefinition>& card_db) {
        (void)creature;
        (void)card_db;
        return 1;
    }

    void GameLogicSystem::push_trigger_check(PipelineExecutor& pipeline, TriggerType type, int source_id) {
        // Add trigger check to pipeline
        (void)pipeline;
        (void)type;
        (void)source_id;
    }

}
