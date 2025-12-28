# Command API Specification

## Overview

The Command API is a type-safe, serialized interface for interacting with the Duel Masters Engine. It replaces the loose dictionary-based `Action` system.

## Command Definition

Commands are defined in C++ (`src/engine/game_command/game_command.hpp`) and exposed to Python via `dm_ai_module.CommandDef` (or `Command` struct).

### CommandType Enum

The `CommandType` enum defines the high-level category of the operation. The following values are strictly synchronized with the C++ engine (`dm_ai_module.CommandType`).

**Core Types (Preferred for new logic):**

| Enum Value | Description |
| :--- | :--- |
| `TRANSITION` | Generic card movement (Move Card). Used for zone changes not covered by specialized types. |
| `MUTATE` | Generic state modification (Tap, Untap, Power, Keywords). |
| `FLOW` | Control flow (Turn/Phase transitions). |
| `QUERY` | Engine request for decision (Select Target). |
| `CHOICE` | Player decision response (Select Option). |

**Specialized Types (Engine Logic):**

| Enum Value | Description |
| :--- | :--- |
| `DRAW_CARD` | Draw card(s). |
| `DISCARD` | Discard card(s). |
| `DESTROY` | Destroy card(s). |
| `MANA_CHARGE` | Place card in mana zone. |
| `TAP` / `UNTAP` | Tap/Untap card (Atomic). |
| `ATTACK_PLAYER` | Attack declaration targeting player. |
| `ATTACK_CREATURE` | Attack declaration targeting creature. |
| `BLOCK` | Block declaration. |
| `BREAK_SHIELD` | Break shield logic. |
| `RESOLVE_BATTLE` | Battle resolution step. |
| `RESOLVE_PLAY` | Card play resolution. |
| `CAST_SPELL` | Spell cast resolution. |
| `SEARCH_DECK` | Search deck effect. |
| `SHUFFLE_DECK` | Shuffle deck effect. |
| `SHIELD_TRIGGER` | Use shield trigger. |
| `MEKRAID` | Mekraid effect. |

**Full Enumeration:**
`TRANSITION`, `MUTATE`, `FLOW`, `QUERY`, `DRAW_CARD`, `DISCARD`, `DESTROY`, `MANA_CHARGE`, `TAP`, `UNTAP`, `POWER_MOD`, `ADD_KEYWORD`, `RETURN_TO_HAND`, `BREAK_SHIELD`, `SEARCH_DECK`, `SHIELD_TRIGGER`, `ATTACK_PLAYER`, `ATTACK_CREATURE`, `BLOCK`, `RESOLVE_BATTLE`, `RESOLVE_PLAY`, `RESOLVE_EFFECT`, `SHUFFLE_DECK`, `LOOK_AND_ADD`, `MEKRAID`, `REVEAL_CARDS`, `PLAY_FROM_ZONE`, `CAST_SPELL`, `SUMMON_TOKEN`, `SHIELD_BURN`, `SELECT_NUMBER`, `CHOICE`, `LOOK_TO_BUFFER`, `SELECT_FROM_BUFFER`, `PLAY_FROM_BUFFER`, `MOVE_BUFFER_TO_ZONE`, `FRIEND_BURST`, `REGISTER_DELAYED_EFFECT`, `NONE`.

### Command Structure (Python `CommandDef`)

The `CommandDef` structure in Python (exposed by `dm_ai_module`) contains the following fields. Not all fields are used for every command type.

| Field | Type | Description |
| :--- | :--- | :--- |
| `type` | `CommandType` | The command type enum. |
| `from_zone` | `Zone` | Source zone for TRANSITION. |
| `to_zone` | `Zone` | Destination zone for TRANSITION/ADD_CARD. |
| `instance_id` | `int` | ID of the card being operated on. |
| `target_instance_id` | `int` | Target instance ID (e.g., for ATTACK/BLOCK). |
| `owner_id` | `int` | Owner player ID. |
| `amount` | `int` | Generic integer value (Power amount, Draw count, etc). |
| `target_group` | `TargetScope` | Scope for targeting (legacy compatibility). |
| `target_filter` | `FilterDef` | Filter for targeting (legacy compatibility). |
| `str_param` | `string` | String parameter (e.g., keyword name, query type). |
| `legacy_warning` | `bool` | Flag indicating a legacy fallback was used. |

## Mapping Rules (Legacy Action -> Command)

To ensure Strict Validation in `CommandSystem`, actions are normalized as follows:

1.  **Destruction**: `DESTROY` (Type: `DESTROY`, to: `GRAVEYARD`) or `TRANSITION` (to `GRAVEYARD`).
2.  **Mana Charge**: `MANA_CHARGE` -> `TRANSITION` (Type: `TRANSITION`, to: `MANA_ZONE`).
3.  **Selection**: `SELECT_TARGET` -> `QUERY` (Type: `QUERY`), `SELECT_OPTION` -> `CHOICE` (Type: `CHOICE`).

## Future Direction (Phase 5+)

- Deprecate `Action` dictionary completely.
- Direct generation of `CommandDef` from agents.
- `CommandSystem` becomes the sole entry point for state mutation.
