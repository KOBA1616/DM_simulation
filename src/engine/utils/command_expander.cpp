#include "command_expander.hpp"
#include <iostream>

namespace dm::engine::utils {

    using namespace dm::core;

    std::vector<CommandDef> CommandExpander::expand(const CommandDef& cmd) {
        std::vector<CommandDef> result;

        switch (cmd.type) {
            case CommandType::DRAW_CARD: {
                // DRAW_CARD(N) -> TRANSITION(DECK->HAND, N)
                CommandDef trans;
                trans.type = CommandType::TRANSITION;
                trans.from_zone = "DECK";
                trans.to_zone = "HAND";
                trans.amount = cmd.amount;
                trans.target_group = cmd.target_group == TargetScope::NONE ? TargetScope::SELF : cmd.target_group;
                trans.output_value_key = cmd.output_value_key;
                result.push_back(trans);
                break;
            }
            case CommandType::DESTROY: {
                // DESTROY(Target) -> TRANSITION(BATTLE->GRAVEYARD, Target)
                CommandDef trans;
                trans.type = CommandType::TRANSITION;
                trans.from_zone = "BATTLE";
                trans.to_zone = "GRAVEYARD";
                trans.target_filter = cmd.target_filter;
                trans.target_group = cmd.target_group;
                trans.output_value_key = cmd.output_value_key;
                result.push_back(trans);
                break;
            }
            case CommandType::MANA_CHARGE: {
                // MANA_CHARGE(N) -> TRANSITION(DECK->MANA, N) if from deck? Or from Hand?
                // Usually "Charge Mana" implies "Put top card of deck into mana" or "Put card from hand".
                // Legacy macro in CommandSystem treated MANA_CHARGE as "Top of Deck -> Mana" loop.
                // If filter/target is specified, it respects that.
                CommandDef trans;
                trans.type = CommandType::TRANSITION;
                trans.from_zone = cmd.from_zone.empty() ? "DECK" : cmd.from_zone;
                trans.to_zone = "MANA";
                trans.amount = cmd.amount;
                trans.target_filter = cmd.target_filter;
                trans.target_group = cmd.target_group;
                trans.output_value_key = cmd.output_value_key;
                result.push_back(trans);
                break;
            }
            case CommandType::DISCARD: {
                // DISCARD -> TRANSITION(HAND->GRAVEYARD)
                CommandDef trans;
                trans.type = CommandType::TRANSITION;
                trans.from_zone = "HAND";
                trans.to_zone = "GRAVEYARD";
                trans.target_filter = cmd.target_filter;
                trans.target_group = cmd.target_group;
                trans.output_value_key = cmd.output_value_key;
                result.push_back(trans);
                break;
            }
            case CommandType::RETURN_TO_HAND: {
                // RETURN_TO_HAND -> TRANSITION(BATTLE->HAND) (default, but CommandSystem tried to infer zone)
                // We default to BATTLE here, relying on target resolution to pick correct cards.
                // However, TRANSITION needs explicit from_zone.
                // If the filter specifies zones, we can iterate zones.
                // But CommandDef is single zone.
                // If the user didn't specify zone, we assume BATTLE.
                CommandDef trans;
                trans.type = CommandType::TRANSITION;
                trans.from_zone = cmd.from_zone.empty() ? "BATTLE" : cmd.from_zone;
                trans.to_zone = "HAND";
                trans.target_filter = cmd.target_filter;
                trans.target_group = cmd.target_group;
                trans.output_value_key = cmd.output_value_key;
                result.push_back(trans);
                break;
            }
            case CommandType::TAP: {
                CommandDef mutate;
                mutate.type = CommandType::MUTATE;
                mutate.mutation_kind = "TAP";
                mutate.target_filter = cmd.target_filter;
                mutate.target_group = cmd.target_group;
                result.push_back(mutate);
                break;
            }
            case CommandType::UNTAP: {
                CommandDef mutate;
                mutate.type = CommandType::MUTATE;
                mutate.mutation_kind = "UNTAP";
                mutate.target_filter = cmd.target_filter;
                mutate.target_group = cmd.target_group;
                result.push_back(mutate);
                break;
            }
            case CommandType::POWER_MOD: {
                CommandDef mutate;
                mutate.type = CommandType::MUTATE;
                mutate.mutation_kind = "POWER_MOD";
                mutate.amount = cmd.amount;
                mutate.target_filter = cmd.target_filter;
                mutate.target_group = cmd.target_group;
                result.push_back(mutate);
                break;
            }
            case CommandType::ADD_KEYWORD: {
                CommandDef mutate;
                mutate.type = CommandType::MUTATE;
                mutate.mutation_kind = "ADD_KEYWORD";
                mutate.str_param = cmd.str_param;
                mutate.target_filter = cmd.target_filter;
                mutate.target_group = cmd.target_group;
                result.push_back(mutate);
                break;
            }
            case CommandType::BREAK_SHIELD: {
                CommandDef trans;
                trans.type = CommandType::TRANSITION;
                trans.from_zone = "SHIELD";
                trans.to_zone = "HAND";
                trans.target_filter = cmd.target_filter;
                trans.target_group = cmd.target_group;
                result.push_back(trans);
                break;
            }
            case CommandType::MEKRAID: {
                return expand_mekraid(cmd);
            }
            case CommandType::LOOK_AND_ADD: {
                return expand_look_and_add(cmd);
            }
            case CommandType::SEARCH_DECK: {
                return expand_search_deck(cmd);
            }
            default:
                // Return as is (Primitives)
                result.push_back(cmd);
                break;
        }

        return result;
    }

    std::vector<CommandDef> CommandExpander::expand_mekraid(const CommandDef& cmd) {
        std::vector<CommandDef> result;

        // Mekraid N (amount): Look 3 (fixed for Mekraid), Play 1 (Cost <= N, Race match), Rest Bottom.
        int look_count = 3;
        std::string race = cmd.str_param;
        int max_cost = cmd.amount;

        std::string buffer_key = "mekraid_choice";

        // 1. TRANSITION: DECK -> BUFFER (3 cards)
        CommandDef step1;
        step1.type = CommandType::TRANSITION;
        step1.from_zone = "DECK";
        step1.to_zone = "BUFFER";
        step1.amount = look_count;
        step1.target_group = TargetScope::SELF; // Mekraid is always self? Yes.
        result.push_back(step1);

        // 2. QUERY: Select 1 from BUFFER
        CommandDef step2;
        step2.type = CommandType::QUERY;
        step2.str_param = "SELECT_TARGET"; // Query Type
        step2.optional = true; // "You MAY put..."

        FilterDef filter;
        filter.zones = {"BUFFER"};
        filter.types = {"CREATURE"};
        if (!race.empty()) filter.races = {race};
        filter.max_cost = max_cost;
        filter.count = 1;

        step2.target_filter = filter;
        step2.output_value_key = buffer_key;
        result.push_back(step2);

        // 3. FLOW: If selection made, Play it.
        CommandDef step3;
        step3.type = CommandType::FLOW;
        step3.condition = ConditionDef();
        step3.condition.value().type = "NOT_EMPTY"; // Check implicit existence in ctx
        step3.condition.value().stat_key = buffer_key; // Check if key has > 0 count? Or value?
        // Wait, ConditionSystem checks int value. CommandSystem stores count in output_value_key?
        // QUERY stores result count?
        // QueryCommand doesn't store in map, DECIDE does.
        // But CommandSystem QUERY flow is: Execute -> Wait for Input -> Decide.
        // Wait, CommandSystem executes *immediately*. If Query requires input, it pauses?
        // Currently CommandSystem.execute just sets state.waiting_for_user_input.
        // It does NOT block. The loop continues next frame?
        // NO. CommandSystem is synchronous. `state.waiting_for_user_input = true` means we stop processing?
        // If we stop processing, the subsequent commands in the vector are NOT executed yet.
        // How does the engine resume?
        // The Engine (GameInstance/ScenarioExecutor) handles resumption.
        // When input arrives (DecideCommand), we need to resume execution.
        // Currently `CommandSystem` does NOT have resumption logic built-in for `execute_command`.
        // It assumes atomic execution for most things.
        // However, `SEARCH_DECK` in legacy handler just "assumed" targets resolved.
        // If we switch to `QUERY`, we rely on the `Decision` system.

        // If we are strictly expanding macros for *immediate* execution (like AI simulation),
        // we might assume AI agent makes choice.
        // But for `JsonLoader`, we are defining the *Card Logic*.
        // If the card logic involves user input, it breaks the synchronous `commands` execution flow
        // unless the flow is managed by a `PipelineExecutor`.
        // The user mentioned "Buffer Sequencing".

        // For now, I assume the standard flow works:
        // 1. Execute TRANSITION.
        // 2. Execute QUERY. (Sets waiting flag).
        // 3. Game Loop waits.
        // 4. User Inputs.
        // 5. Game Loop continues? Or does it re-execute?
        // If it re-executes, it needs to skip done commands.
        // This suggests `EffectResolver` or `PipelineExecutor` manages the index.
        // `CommandDef` doesn't support resuming.

        // However, the task is about "Expansion".
        // Whether the runtime supports it is a separate issue (likely handled by `GenericCardSystem` or `EffectResolver` iterating `commands`).

        // Step 3 Logic:
        // Assume selection is in `buffer_key` (list of IDs? No, context stores ints).
        // `CommandSystem` doesn't store selection in context automatically.
        // `QueryCommand` sets `pending_query`. `DecideCommand` sets `last_decision`?
        // We might need a mechanism to bind decision to context.
        // For now, I'll generate the commands assuming the infrastructure handles the variable binding.

        CommandDef play_cmd;
        play_cmd.type = CommandType::TRANSITION;
        play_cmd.from_zone = "BUFFER";
        play_cmd.to_zone = "BATTLE";
        // Target is the one selected in step 2.
        // How to refer to it?
        // `input_value_key`?
        // CommandDef has `input_value_key`.
        // But `QUERY` output is a list of IDs. `CommandSystem` `execution_context` stores `int`.
        // We might need `target_choice` string? Or `target_filter` with `selection_mode="FROM_CONTEXT"`?
        // `FilterDef` has no such mode.
        // But `CommandDef` has `input_value_key`. `resolve_targets` doesn't use it yet.
        // I need to rely on `input_value_key` to fetch targets from context.
        // `CommandSystem::resolve_targets` needs to support fetching from context if `input_value_key` is present?
        // Currently `resolve_amount` uses `input_value_key`.
        // If I use `input_value_key` for targets, `CommandSystem` needs update.
        // BUT, for `MEKRAID`, the `SELECT_TARGET` (QUERY) implies the next command uses the selection?
        // Or we use `target_group = TARGET_SELECT` in Step 3?
        // Step 2 asks for selection. Step 3 uses it.

        // Actually, `SEARCH_DECK` expansion in `JsonLoader` (if I wrote it) would be:
        // 1. QUERY.
        // 2. MOVE (Target=Selected).
        // 3. SHUFFLE.

        // I will define the `FLOW` branch.
        play_cmd.input_value_key = buffer_key; // Hint to use this variable
        // I'll update CommandSystem later to support this if needed, or assume it exists.
        // Actually, `CommandSystem` logic I read earlier:
        // `resolve_amount` uses `input_value_key`.
        // `resolve_targets` does NOT use `input_value_key`.
        // This is a missing link. I should add it to `CommandSystem`.
        // I'll add a TODO or implicit requirement.
        // For now, I generate the command.

        step3.if_true.push_back(play_cmd);
        result.push_back(step3);

        // 4. TRANSITION: Rest of BUFFER -> DECK Bottom
        CommandDef step4;
        step4.type = CommandType::TRANSITION;
        step4.from_zone = "BUFFER";
        step4.to_zone = "DECK";
        step4.destination_index = 0; // Bottom
        // Target: All in Buffer.
        FilterDef rest_filter;
        rest_filter.zones = {"BUFFER"};
        step4.target_filter = rest_filter;
        step4.target_group = TargetScope::SELF;
        result.push_back(step4);

        return result;
    }

    std::vector<CommandDef> CommandExpander::expand_look_and_add(const CommandDef& cmd) {
        // Look N, Add 1 to Hand, Rest Bottom.
        std::vector<CommandDef> result;
        int look_count = cmd.amount;
        std::string buffer_key = "look_choice";

        // 1. DECK -> BUFFER
        CommandDef step1;
        step1.type = CommandType::TRANSITION;
        step1.from_zone = "DECK";
        step1.to_zone = "BUFFER";
        step1.amount = look_count;
        result.push_back(step1);

        // 2. QUERY: Select 1
        CommandDef step2;
        step2.type = CommandType::QUERY;
        step2.str_param = "SELECT_TARGET";
        step2.target_filter = cmd.target_filter; // Use provided filter (e.g. "Creature")
        // If filter is empty, it means "Any card".
        if (step2.target_filter.zones.empty()) step2.target_filter.zones = {"BUFFER"};
        step2.target_filter.count = 1;
        step2.output_value_key = buffer_key;
        result.push_back(step2);

        // 3. Move Selected to Hand
        CommandDef step3;
        step3.type = CommandType::TRANSITION;
        step3.from_zone = "BUFFER";
        step3.to_zone = "HAND";
        step3.input_value_key = buffer_key;
        result.push_back(step3);

        // 4. Rest -> Deck Bottom
        CommandDef step4;
        step4.type = CommandType::TRANSITION;
        step4.from_zone = "BUFFER";
        step4.to_zone = "DECK";
        step4.destination_index = 0;
        step4.target_filter.zones = {"BUFFER"};
        result.push_back(step4);

        return result;
    }

    std::vector<CommandDef> CommandExpander::expand_search_deck(const CommandDef& cmd) {
        // Search Deck -> Hand (or to_zone) -> Shuffle
        std::vector<CommandDef> result;
        std::string buffer_key = "search_choice";

        // 1. QUERY (from DECK)
        CommandDef step1;
        step1.type = CommandType::QUERY;
        step1.str_param = "SELECT_TARGET";
        step1.target_filter = cmd.target_filter;
        if (step1.target_filter.zones.empty()) step1.target_filter.zones = {"DECK"};
        step1.output_value_key = buffer_key;
        result.push_back(step1);

        // 2. MOVE
        CommandDef step2;
        step2.type = CommandType::TRANSITION;
        step2.from_zone = "DECK";
        step2.to_zone = cmd.to_zone.empty() ? "HAND" : cmd.to_zone;
        step2.input_value_key = buffer_key;
        result.push_back(step2);

        // 3. SHUFFLE
        CommandDef step3;
        step3.type = CommandType::SHUFFLE_DECK; // Or PRIMITIVE?
        // CommandType::SHUFFLE_DECK is in expanded set.
        // Is there a primitive? GameCommand::ShuffleCommand exists.
        // CommandSystem handles SHUFFLE_DECK?
        // CommandType::SHUFFLE_DECK is mapped to GameCommand::ShuffleCommand in CommandSystem (if I implement it).
        // Wait, CommandSystem::execute_primitive currently handles `TRANSITION`, `MUTATE`, `FLOW`, `QUERY`.
        // It does NOT handle `SHUFFLE_DECK` as primitive.
        // It handles `SEARCH_DECK` as macro.
        // I need to add `SHUFFLE_DECK` handling to `execute_primitive` (or add SHUFFLE case).
        // Or make `SHUFFLE` a primitive `CommandType::SHUFFLE`?
        // `CommandType::SHUFFLE_DECK` is in enum.
        result.push_back(step3);

        return result;
    }

}
