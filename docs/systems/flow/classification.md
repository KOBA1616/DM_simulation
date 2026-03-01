> **Legacy Terminology Note:**
> This document uses the term "Action" to describe player moves and game state transitions.
> In the current implementation, these are implemented as `CommandDef` structures within the Command System.
> The high-level logic described here remains valid, but "Action" should be interpreted as "Command" or "Player Intent".

```mermaid
graph TB
    subgraph "ユーザーアクション (15種類)"
        subgraph "フェーズアクション"
            MANA[MANA_CHARGE<br/>マナチャージ<br/>優先度: 90 MANA]
            PLAY[DECLARE_PLAY<br/>カードプレイ<br/>優先度: 80 MAIN]
            ATK_P[ATTACK_PLAYER<br/>プレイヤー攻撃<br/>優先度: 85 ATTACK]
            ATK_C[ATTACK_CREATURE<br/>クリーチャー攻撃<br/>優先度: 85 ATTACK]
            BLOCK[BLOCK<br/>ブロック宣言<br/>優先度: 85 BLOCK]
            PASS[PASS<br/>パス<br/>優先度: 0]
        end
        
        subgraph "クエリ応答"
            SEL_T[SELECT_TARGET<br/>ターゲット選択<br/>優先度: 100]
            SEL_O[SELECT_OPTION<br/>オプション選択<br/>優先度: 100]
            SEL_N[SELECT_NUMBER<br/>数値選択<br/>優先度: 100]
        end
        
        subgraph "効果関連"
            RES_E[RESOLVE_EFFECT<br/>効果解決<br/>優先度: 100/95]
            USE_ST[USE_SHIELD_TRIGGER<br/>S・トリガー使用<br/>優先度: 50]
            USE_AB[USE_ABILITY<br/>能力使用<br/>優先度: 50]
            DEC_R[DECLARE_REACTION<br/>リアクション宣言<br/>優先度: 50]
        end
        
        subgraph "レガシー/非推奨"
            MOV[MOVE_CARD<br/>カード移動<br/>非推奨]
            PLAY_OLD[PLAY_CARD<br/>カードプレイ旧<br/>非推奨]
            PLAY_Z[PLAY_FROM_ZONE<br/>ゾーンからプレイ<br/>特殊]
        end
    end
    
    subgraph "エンジン内部アクション (7種類)"
        subgraph "Atomicフロー"
            DEC_P[DECLARE_PLAY<br/>プレイ宣言<br/>優先度: 80]
            PAY[PAY_COST<br/>コスト支払い<br/>優先度: 98]
            RES_P[RESOLVE_PLAY<br/>プレイ解決<br/>優先度: 98]
        end
        
        subgraph "バトル/効果"
            RES_B[RESOLVE_BATTLE<br/>バトル解決<br/>優先度: 100]
            BRK[BREAK_SHIELD<br/>シールドブレイク<br/>優先度: 100]
            PLAY_I[PLAY_CARD_INTERNAL<br/>内部プレイ<br/>優先度: 80]
        end
    end
    
    DEC_P --> PAY --> RES_P
    RES_P --> RES_E
    ATK_P --> RES_B
    ATK_C --> RES_B
    RES_B --> BRK
    
    style SEL_T fill:#ff9999
    style SEL_O fill:#ff9999
    style SEL_N fill:#ff9999
    style RES_E fill:#ffcc99
    style PAY fill:#ffff99
    style RES_P fill:#ffff99
    style RES_B fill:#ff9999
    style BRK fill:#ff9999
    style MANA fill:#99ccff
    style PLAY fill:#99ccff
    style ATK_P fill:#99ccff
    style ATK_C fill:#99ccff
    style BLOCK fill:#99ccff
    style PASS fill:#cccccc
    style MOV fill:#dddddd
    style PLAY_OLD fill:#dddddd
```
