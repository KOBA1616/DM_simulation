> **Legacy Terminology Note:**
> This document uses the term "Action" to describe player moves and game state transitions.
> In the current implementation, these are implemented as `CommandDef` structures within the Command System.
> The high-level logic described here remains valid, but "Action" should be interpreted as "Command" or "Player Intent".

```mermaid
stateDiagram-v2
    [*] --> CheckState: generate_legal_actions()
    
    CheckState --> QueryWait: waiting_for_user_input == true
    CheckState --> PendingCheck: waiting_for_user_input == false
    
    QueryWait --> ReturnQuery: 優先度100<br/>SELECT_TARGET<br/>SELECT_OPTION
    ReturnQuery --> [*]: クエリ応答完了まで<br/>他アクションブロック
    
    PendingCheck --> PendingResolve: !pending_effects.empty()
    PendingCheck --> StackCheck: pending_effects.empty()
    
    PendingResolve --> ReturnResolve: 優先度100/95<br/>RESOLVE_EFFECT<br/>SKIP_EFFECT
    ReturnResolve --> [*]: 効果解決完了まで<br/>他アクションブロック
    
    StackCheck --> StackProcess: !stack.empty()
    StackCheck --> PhaseAction: stack.empty()
    
    StackProcess --> ReturnStack: 優先度98<br/>PAY_COST<br/>RESOLVE_PLAY
    ReturnStack --> [*]: スタック処理完了まで<br/>他アクションブロック
    
    PhaseAction --> ManaPhase: phase == MANA
    PhaseAction --> MainPhase: phase == MAIN
    PhaseAction --> AttackPhase: phase == ATTACK
    PhaseAction --> BlockPhase: phase == BLOCK
    PhaseAction --> AutoPhase: phase == START/DRAW/END
    
    ManaPhase --> ReturnMana: 優先度90<br/>MANA_CHARGE<br/>PASS
    MainPhase --> ReturnMain: 優先度80<br/>DECLARE_PLAY<br/>PASS
    AttackPhase --> ReturnAttack: 優先度85<br/>ATTACK<br/>PASS
    BlockPhase --> ReturnBlock: 優先度85<br/>BLOCK<br/>PASS
    AutoPhase --> ReturnEmpty: 空リスト<br/>fast_forward
    
    ReturnMana --> [*]
    ReturnMain --> [*]
    ReturnAttack --> [*]
    ReturnBlock --> [*]
    ReturnEmpty --> [*]
    
    note right of QueryWait
        Level 1: 最優先
        他のすべてのアクションをマスク
    end note
    
    note right of PendingResolve
        Level 2: 非常に高優先度
        フェーズアクションをマスク
        必須効果: 100
        オプショナル効果: 95
    end note
    
    note right of StackProcess
        Level 3: 高優先度
        フェーズアクションをマスク
    end note
    
    note right of PhaseAction
        Level 4: フェーズ依存
        フェーズごとに適切な
        アクションのみ生成
    end note
```
