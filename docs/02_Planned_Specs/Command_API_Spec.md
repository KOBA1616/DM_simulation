# Command API Specification

## Overview

This document defines the schema and protocol for the new `Command` system, which replaces the legacy `Action` dictionary structure. The goal is to provide a strongly-typed, consistent, and serializable representation of game actions.

## ICommand Protocol

The `ICommand` protocol (defined in `dm_toolkit.commands_new`) mandates the following interface for all command objects:

```python
class ICommand(Protocol):
    def execute(self, state: Any) -> Optional[Any]:
        """
        Executes the command on the given game state.
        Returns the result of execution (e.g., True/False, or modified state).
        """
        ...

    def invert(self, state: Any) -> Optional[Any]:
        """
        Returns an inverse command or executes the inverse operation, if applicable.
        Used for undoing actions in search trees or debugging.
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the command, adhering to the Command Schema.
        """
        ...
```

## Command Schema (JSON)

The `to_dict()` method must return a dictionary matching the following structure.

### Core Fields

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `type` | String | Yes | The command type (e.g., `PLAY_CARD`, `ATTACK_PLAYER`). Should match `CommandType` enum values if possible. |
| `uid` | String | Yes | Unique identifier for the command instance (UUID v4). |
| `from_zone` | String | No | Source zone (e.g., `HAND`, `BATTLE_ZONE`). |
| `to_zone` | String | No | Destination zone (e.g., `GRAVEYARD`, `MANA_ZONE`). |
| `instance_id` | Integer | No | ID of the primary card/object being acted upon. |
| `target_player` | String/Int | No | Target player ID or alias (`SELF`, `OPPONENT`). |
| `amount` | Integer | No | Numeric value associated with the action (e.g., mana cost, power, count). |
| `options` | List[List[Dict]] | No | Nested commands for choices. Outer list = choices, Inner list = sequence of commands for that choice. |
| `flags` | List[String] | No | Boolean flags (e.g., `OPTIONAL`, `ALLOW_DUPLICATES`). |
| `legacy_warning` | Boolean | No | If true, indicates this command was imperfectly mapped from a legacy Action. |

### Extended Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `target_group` | String | Defines the scope of targeting (e.g., `TARGET_SELECT`, `ALL`). |
| `target_filter` | Dict | A `FilterDef` dictionary defining valid targets. |
| `input_value_key` | String | Key to retrieve input value from execution context. |
| `output_value_key` | String | Key to store output value in execution context. |
| `str_param` | String | Generic string parameter (e.g., for query types). |
| `mutation_kind` | String | Specific kind of mutation for `MUTATE` commands. |

### Example

```json
{
  "type": "PLAY_CARD",
  "uid": "123e4567-e89b-12d3-a456-426614174000",
  "from_zone": "HAND",
  "to_zone": "BATTLE_ZONE",
  "instance_id": 101,
  "target_player": "SELF",
  "amount": 5,
  "flags": ["OPTIONAL"]
}
```

## Mapping Rules (Legacy Action -> Command)

The `ActionToCommandMapper` is responsible for converting legacy dictionaries. Key transformation rules include:

1.  **Zone Normalization**: Keys like `source_zone`, `origin_zone` mapped to `from_zone`. `destination_zone`, `dest_zone` mapped to `to_zone`.
2.  **Type Mapping**: Legacy types like `MANA_CHARGE` are mapped to `TRANSITION` (or specific `MANA_CHARGE` type if supported) with `to_zone="MANA_ZONE"`.
3.  **Recursion**: `options` containing lists of actions are recursively mapped to lists of commands.

## C++ Interop

Python `ICommand` implementations may wrap C++ `GameCommand` objects. In such cases:
- `execute()` delegates to the C++ `execute()` method.
- `to_dict()` provides a Python-side reconstruction or delegates to a C++ serialization method if available.
