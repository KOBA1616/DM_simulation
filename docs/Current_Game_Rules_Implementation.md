# Current Game Rules & Implementation Status (v2.2)

This document outlines the game rules currently implemented in the C++ engine (`dm::engine`) and how they map to the source code.

## 1. Game Flow & Turn Structure
**Implementation**: `src/engine/flow/phase_manager.cpp`

The game follows a strict phase sequence managed by `PhaseManager`.

### Phase Sequence
1.  **START_OF_TURN**
    *   **Rules**:
        *   Active Player's mana and creatures untap (`ManaSystem::untap_all`).
        *   Summoning Sickness is removed from Active Player's creatures.
        *   `AT_START_OF_TURN` effects are queued.
    *   **Transition**: Automatically proceeds to `DRAW`.

2.  **DRAW**
    *   **Rules**:
        *   Active Player draws 1 card.
        *   **Exception**: Player 1 skips draw on the very first turn (Turn 1).
        *   **Loss Condition**: If deck is empty when drawing, player loses (`check_game_over`).
    *   **Transition**: Automatically proceeds to `MANA`.

3.  **MANA**
    *   **Rules**:
        *   Player *may* charge 1 card from hand to Mana Zone.
        *   **Implementation**: `ActionGenerator` generates `MANA_CHARGE` actions for all cards in hand, plus a `PASS` action.
    *   **Transition**: Ends when player performs an action (Charge or Pass). Proceeds to `MAIN`.

4.  **MAIN**
    *   **Rules**:
        *   Player may summon creatures or cast spells.
        *   **Cost System** (`ManaSystem::can_pay_cost`):
            *   Total Cost <= Number of Untapped Mana.
            *   At least 1 mana of the card's civilization must be included (unless Zero civ).
    *   **Transition**: Ends when player chooses `PASS`. Proceeds to `ATTACK`.

5.  **ATTACK**
    *   **Rules**:
        *   Player may attack with any creature that is **Untapped** and does **not have Summoning Sickness**.
        *   **Targets**: Opponent Player or Opponent's **Tapped** Creature.
        *   **Multiple Attacks**: There is **NO restriction** on the number of attacks per turn. The phase continues until the player chooses `PASS` or runs out of attackers.
        *   **Tapping**: Attacking causes the creature to Tap (`EffectResolver::resolve_attack`).
    *   **Transition**:
        *   If `ATTACK_PLAYER` or `ATTACK_CREATURE` is chosen -> Proceeds to `BLOCK`.
        *   If `PASS` is chosen -> Proceeds to `END_OF_TURN`.

6.  **BLOCK**
    *   **Rules**:
        *   Non-Active Player (Defender) may block with a creature having `BLOCKER` keyword.
        *   Blocker must be Untapped.
        *   Blocking Taps the blocker.
    *   **Transition**:
        *   After Block (or Pass), Battle is resolved (`EffectResolver::execute_battle`).
        *   Game returns to `ATTACK` phase.

7.  **END_OF_TURN**
    *   **Rules**:
        *   `AT_END_OF_TURN` effects are queued.
        *   Active Player switches.
    *   **Transition**: Proceeds to `START_OF_TURN`.

## 2. Battle Logic
**Implementation**: `src/engine/effects/effect_resolver.cpp`

### Battle Resolution (`execute_battle`)
1.  **Power Calculation**:
    *   Base Power is retrieved from `CardDefinition`.
    *   **Power Attacker**: If attacking, `power_attacker_bonus` is added.
2.  **Outcome**:
    *   Attacker Power > Defender Power -> Defender destroyed.
    *   Attacker Power < Defender Power -> Attacker destroyed.
    *   Attacker Power = Defender Power -> Both destroyed.
3.  **Special Abilities**:
    *   **Slayer**: If a Slayer loses or draws, the winner is also destroyed.
4.  **Shield Break** (If attacking player and unblocked):
    *   **Breaker Count**:
        *   Default: 1
        *   `DOUBLE_BREAKER`: 2
        *   `TRIPLE_BREAKER`: 3
    *   Shields are moved to Hand.
    *   **Shield Trigger**: If a shield has `SHIELD_TRIGGER`, it can be played immediately for free (`resolve_use_shield_trigger`).

## 3. Card Abilities (Keywords)
**Implementation**: `src/core/card_def.hpp`, `src/engine/effects/effect_resolver.cpp`

*   **SPEED_ATTACKER**: Ignores Summoning Sickness.
*   **BLOCKER**: Can redirect attacks to self.
*   **SLAYER**: Destroys opponent in battle regardless of power.
*   **POWER_ATTACKER**: Gains power during attack.
*   **DOUBLE/TRIPLE_BREAKER**: Breaks extra shields.
*   **SHIELD_TRIGGER**: Casts for free when broken.
*   **EVOLUTION**: Can be placed on top of another creature (Logic partially implemented in `resolve_play_card`).

## 4. Win Conditions
**Implementation**: `src/engine/flow/phase_manager.cpp`

1.  **Direct Attack**: Attacking player with 0 shields.
2.  **Deck Out**: Drawing from an empty deck.
3.  **Turn Limit**: Reaching 100 turns (Result: Draw).

## 5. AI & Automation
*   **Action Generation**: `ActionGenerator` enumerates all legal moves based on the rules above.
*   **Resolution**: `EffectResolver` applies the changes deterministically.
*   **MCTS**: Explores the game tree using these rules.

---
**Note on "1 Turn 1 Attack"**:
The engine explicitly supports multiple attacks. The `ATTACK` phase is a loop. After an attack resolves (and potential block/battle), the state returns to `ATTACK` phase, and `ActionGenerator` will generate attack actions for any remaining untaped creatures. The turn only ends when the AI (or player) explicitly chooses `PASS`.
