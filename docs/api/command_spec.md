# Command API Specification

## Overview

The Command API is a type-safe, serialized interface for interacting with the Duel Masters Engine. It replaces the loose dictionary-based `Action` system.

## Command Definition

Commands are defined in C++ (`src/engine/game_command/game_command.hpp`) and exposed to Python via `dm_ai_module.CommandDef` (or `Command` struct).

### CommandType Enum

The `CommandType` enum defines the high-level category of the operation.

| Enum Value | Description |
| :--- | :--- |
| `TRANSITION` | Moves a card between zones (Hand, Deck, Mana, Battle, Grave, Shield). |
| `MUTATE` | Modifies card state (Tap, Untap, Power, Keywords) or Game State (Phase). |
| `ATTACH` | Attaches a card to another (Evolution, Cross Gear). |
| `FLOW` | Controls game flow (Phase change, Turn change, Attack declaration). |
| `QUERY` | Requests a decision from a player (Target selection, Effect choice). |
| `DECIDE` | The player's response to a QUERY. |
| `DECLARE_REACTION` | Declares a reaction (Shield Trigger, Ninja Strike, etc). |
| `STAT` | Updates game statistics (Cards drawn, etc). |
| `GAME_RESULT` | Sets the game result (Win/Loss). |
| `ADD_CARD` | Adds a new card instance to the game (Token summoning). |
| `SHUFFLE` | Shuffles a player's deck. |

### Command Structure (Python `CommandDef`)

The `CommandDef` structure in Python (exposed by `dm_ai_module`) contains the following fields. Note that not all fields are used for every command type.

| Field | Type | Description |
| :--- | :--- | :--- |
| `type` | `CommandType` | The command type enum. |
| `from_zone` | `Zone` | Source zone for TRANSITION. |
| `to_zone` | `Zone` | Destination zone for TRANSITION/ADD_CARD. |
| `instance_id` | `int` | ID of the card being operated on. |
| `owner_id` | `int` | Owner player ID. |
| `amount` | `int` | Generic integer value (Power amount, Draw count, etc). |
| `target_group` | `TargetScope` | Scope for targeting (legacy compatibility). |
| `target_filter` | `FilterDef` | Filter for targeting (legacy compatibility). |
| `target_instance` | `int` | Target instance ID (e.g., for ATTACK/BLOCK). |
| `str_param` | `string` | String parameter (e.g., keyword name, query type). |
| `legacy_warning` | `bool` | Flag indicating a legacy fallback was used. |

## Mapping Rules (Legacy Action -> Command)

### 1. MOVE_CARD / Zone Changes
Maps to `TRANSITION` or specific types like `MANA_CHARGE`.

| Legacy Action | Command Type | Notes |
| :--- | :--- | :--- |
| `MOVE_CARD` (to Grave) | `TRANSITION` (to `GRAVEYARD`) | "DESTROY" or "DISCARD" based on source. |
| `MOVE_CARD` (to Mana) | `TRANSITION` (to `MANA_ZONE`) | "MANA_CHARGE". |
| `MOVE_CARD` (to Hand) | `TRANSITION` (to `HAND`) | "RETURN_TO_HAND". |
| `DRAW_CARD` | `TRANSITION` (Deck -> Hand) | Uses `amount` for count. |

### 2. State Mutation
Maps to `MUTATE` or `FLOW`.

| Legacy Action | Command Type | Notes |
| :--- | :--- | :--- |
| `TAP` / `UNTAP` | `MUTATE` (Type: `TAP`/`UNTAP`) | |
| `APPLY_MODIFIER` | `MUTATE` (Type: `ADD_COST_MODIFIER` etc) | |
| `GRANT_KEYWORD` | `MUTATE` (Type: `ADD_KEYWORD`) | |

### 3. Game Flow
Maps to `FLOW`.

| Legacy Action | Command Type | Notes |
| :--- | :--- | :--- |
| `ATTACK_PLAYER` | `FLOW` (Type: `SET_ATTACK_PLAYER`) | |
| `ATTACK_CREATURE` | `FLOW` (Type: `SET_ATTACK_TARGET`) | |
| `BLOCK` | `FLOW` (Type: `BLOCK`) | |

### 4. Queries / Selection
Maps to `QUERY` or `DECIDE` depending on context (usually `QUERY` for engine-generated requests).

| Legacy Action | Command Type | Notes |
| :--- | :--- | :--- |
| `SELECT_TARGET` | `QUERY` | |
| `SELECT_OPTION` | `QUERY` (Type: `CHOICE`) | |

## Transition Strategy

1. **Refactor Mapper**: Create `dm_toolkit.action_to_command` to implement pure conversion logic.
2. **Shim**: Use `wrap_action` in `dm_toolkit.commands_new` to transparently convert legacy Actions to Commands using the mapper.
3. **Migrate GUI**: Update `zone_widget.py` to optionally emit Commands.
