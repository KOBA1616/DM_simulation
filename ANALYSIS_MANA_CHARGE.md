# Mana Charge Failure Analysis

## Summary
The "Mana Charge" action in the native C++ engine fails to register correctly in the game state, specifically failing to set the `mana_charged_by_player` flag. This prevents the `PhaseSystem` and `IntentGenerator` from recognizing the action as completed.

## Root Cause Analysis
The issue stems from a discrepancy in how the `MANA_CHARGE` command is processed in `CommandSystem::generate_instructions` versus `GameLogicSystem::dispatch_command`.

### Correct Path (GameLogicSystem)
In `src/engine/systems/director/game_logic_system.cpp`, `MANA_CHARGE` is handled by creating a `GAME_ACTION` instruction:
```cpp
            case core::CommandType::MANA_CHARGE:
            {
                nlohmann::json args;
                args["card"] = cmd.instance_id;
                Instruction inst(InstructionOp::GAME_ACTION, args);
                inst.args["type"] = "MANA_CHARGE";
                handle_mana_charge(pipeline, state, inst); // Calls PlaySystem::handle_mana_charge
                // ...
            }
```
`PlaySystem::handle_mana_charge` explicitly sets the flag:
```cpp
         auto flow_cmd = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_MANA_CHARGED, 1);
         state.execute_command(std::move(flow_cmd));
```

### Incorrect Path (CommandSystem - Current Implementation)
In `src/engine/infrastructure/commands/command_system.cpp`, `MANA_CHARGE` is converted directly to a raw `MOVE` instruction:
```cpp
            case core::CommandType::MANA_CHARGE:
                {
                    Instruction move(InstructionOp::MOVE);
                    int target = (cmd.instance_id != -1) ? cmd.instance_id : source_instance_id;
                    move.args["target"] = target;
                    move.args["to"] = "MANA";
                    out.push_back(move);
                }
                break;
```
This raw `MOVE` instruction bypasses `PlaySystem::handle_mana_charge`, causing the `SET_MANA_CHARGED` flag update to be skipped entirely.

## Verified Reproduction
Using a headless reproduction script (`reproduce_mana_charge.py`) confirmed that the Python fallback shim (`dm_ai_module.py`) correctly implements the logic (moves card + updates state), but the native implementation path identified above is flawed.

## Solution Proposal
Modify `src/engine/infrastructure/commands/command_system.cpp` to align with `GameLogicSystem`.

**Change:**
Replace the `MOVE` instruction generation with a `GAME_ACTION` instruction:

```cpp
            case core::CommandType::MANA_CHARGE:
                {
                    nlohmann::json args;
                    int target = (cmd.instance_id != -1) ? cmd.instance_id : source_instance_id;
                    args["card"] = target;

                    Instruction inst(InstructionOp::GAME_ACTION, args);
                    inst.args["type"] = "MANA_CHARGE";
                    out.push_back(inst);
                }
                break;
```
This change will ensure the command routes through the correct logic handler, setting the necessary game state flags.
