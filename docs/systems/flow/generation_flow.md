> **Legacy Terminology Note:**
> This document uses the term "Action" to describe player moves and game state transitions.
> In the current implementation, these are implemented as `CommandDef` structures within the Command System.
> The high-level logic described here remains valid, but "Action" should be interpreted as "Command" or "Player Intent".

```mermaid
flowchart LR
    subgraph "MANA Phase"
        M1[MANA_CHARGE 優先度90]
        M2[PASS 優先度0]
    end
    
    subgraph "MAIN Phase"
        P1[DECLARE_PLAY 優先度80]
        P2[PLAY_CARD_INTERNAL 優先度80]
        P3[PASS 優先度0]
    end
    
    subgraph "ATTACK Phase"
        A1[ATTACK_PLAYER 優先度85]
        A2[ATTACK_CREATURE 優先度85]
        A3[PASS 優先度0]
    end
    
    subgraph "BLOCK Phase"
        B1[BLOCK 優先度85]
        B2[PASS 優先度0]
    end
    
    subgraph "Stack Processing"
        S1[PAY_COST 優先度98]
        S2[RESOLVE_PLAY 優先度98]
    end
    
    subgraph "Pending Effects"
        E1[RESOLVE_EFFECT 必須 優先度100]
        E2[RESOLVE_EFFECT オプション 優先度95]
        E3[SELECT_TARGET 優先度100]
        E4[USE_SHIELD_TRIGGER 優先度50]
        E5[RESOLVE_BATTLE 優先度100]
        E6[BREAK_SHIELD 優先度100]
    end
    
    subgraph "Query Response"
        Q1[SELECT_TARGET 優先度100]
        Q2[SELECT_OPTION 優先度100]
        Q3[SELECT_NUMBER 優先度100]
    end
    
    Query[waiting_for_user_input] ==>|Level 1| Q1
    Query ==>|Level 1| Q2
    Query ==>|Level 1| Q3
    
    Pending[!pending_effects.empty] ==>|Level 2| E1
    Pending ==>|Level 2| E2
    Pending ==>|Level 2| E3
    Pending ==>|Level 2| E5
    Pending ==>|Level 2| E6
    
    Stack[!stack.empty] ==>|Level 3| S1
    Stack ==>|Level 3| S2
    
    ManaP[Phase==MANA] ==>|Level 4| M1
    MainP[Phase==MAIN] ==>|Level 4| P1
    AttackP[Phase==ATTACK] ==>|Level 4| A1
    AttackP ==>|Level 4| A2
    BlockP[Phase==BLOCK] ==>|Level 4| B1
    
    style Q1 fill:#ff9999,stroke:#cc0000,stroke-width:3px
    style Q2 fill:#ff9999,stroke:#cc0000,stroke-width:3px
    style Q3 fill:#ff9999,stroke:#cc0000,stroke-width:3px
    style E1 fill:#ffcc99,stroke:#ff6600,stroke-width:3px
    style E2 fill:#ffcc99,stroke:#ff6600,stroke-width:2px
    style E5 fill:#ffcc99,stroke:#ff6600,stroke-width:3px
    style E6 fill:#ffcc99,stroke:#ff6600,stroke-width:3px
    style S1 fill:#ffff99,stroke:#cccc00,stroke-width:2px
    style S2 fill:#ffff99,stroke:#cccc00,stroke-width:2px
    style M1 fill:#99ccff,stroke:#0066cc,stroke-width:2px
    style P1 fill:#99ccff,stroke:#0066cc,stroke-width:2px
    style A1 fill:#99ccff,stroke:#0066cc,stroke-width:2px
    style A2 fill:#99ccff,stroke:#0066cc,stroke-width:2px
    style B1 fill:#99ccff,stroke:#0066cc,stroke-width:2px
```
