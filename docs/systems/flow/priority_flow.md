> **Legacy Terminology Note:**
> This document uses the term "Action" to describe player moves and game state transitions.
> In the current implementation, these are implemented as `CommandDef` structures within the Command System.
> The high-level logic described here remains valid, but "Action" should be interpreted as "Command" or "Player Intent".

```mermaid
graph TD
    Start[ゲーム進行] --> CheckQuery{waiting_for_user_input?}
    
    CheckQuery -->|YES Level 1| QueryActions[クエリ応答アクション生成]
    QueryActions --> GenTarget[SELECT_TARGET優先度100]
    QueryActions --> GenOption[SELECT_OPTION優先度100]
    
    CheckQuery -->|NO| CheckPending{pending_effects.empty?}
    
    CheckPending -->|NO Level 2| PendingActions[エフェクト解決アクション生成]
    PendingActions --> CheckOptional{optional?}
    CheckOptional -->|NO| GenResolve[RESOLVE_EFFECT優先度100必須]
    CheckOptional -->|YES| GenResolveOpt[RESOLVE_EFFECT優先度95]
    CheckOptional -->|YES| GenSkip[SKIP_EFFECT優先度50]
    
    CheckPending -->|YES| CheckStack{stack.empty?}
    
    CheckStack -->|NO Level 3| StackActions[スタック処理アクション生成]
    StackActions --> GenPay[PAY_COST優先度98]
    StackActions --> GenResPlay[RESOLVE_PLAY優先度98]
    
    CheckStack -->|YES| PhaseCheck{current_phase}
    
    PhaseCheck -->|MANA| ManaActions[マナフェーズアクション]
    ManaActions --> GenMana[MANA_CHARGE優先度90]
    ManaActions --> GenPassM[PASS優先度0]
    
    PhaseCheck -->|MAIN| MainActions[メインフェーズアクション]
    MainActions --> GenPlay[DECLARE_PLAY優先度80]
    MainActions --> GenPassMa[PASS優先度0]
    
    PhaseCheck -->|ATTACK| AttackActions[攻撃フェーズアクション]
    AttackActions --> GenAttack[ATTACK優先度85]
    AttackActions --> GenPassA[PASS優先度0]
    
    PhaseCheck -->|BLOCK| BlockActions[ブロックフェーズアクション]
    BlockActions --> GenBlock[BLOCK優先度85]
    BlockActions --> GenPassB[PASS優先度0]
    
    PhaseCheck -->|START/DRAW/END| AutoAdvance[空リスト自動進行]
    
    GenTarget --> AISelect[SimpleAI::select_action]
    GenOption --> AISelect
    GenResolve --> AISelect
    GenResolveOpt --> AISelect
    GenSkip --> AISelect
    GenPay --> AISelect
    GenResPlay --> AISelect
    GenMana --> AISelect
    GenPlay --> AISelect
    GenAttack --> AISelect
    GenBlock --> AISelect
    GenPassM --> AISelect
    GenPassMa --> AISelect
    GenPassA --> AISelect
    GenPassB --> AISelect
    
    AISelect --> Execute[アクション実行]
    AutoAdvance --> FastForward[fast_forward]
    
    style CheckQuery fill:#ff9999
    style CheckPending fill:#ffcc99
    style CheckStack fill:#ffff99
    style PhaseCheck fill:#99ccff
    style AISelect fill:#99ff99
```
