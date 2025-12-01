# AI学習戦略設計書
## ～ カリキュラム学習とメタゲーム・リーグによる汎用性獲得 ～

## 1. 概要
本ドキュメントは、特定のデッキにしか勝てない「特化型AI」ではなく、あらゆる戦況に対応できる「汎用型AI」を育成するための学習プロセスを定義する。

主なアプローチ:
- **Dual Curriculum (二極化カリキュラム)**: 「アグロ」と「コントロール」という正反対のデッキを交互に使用させ、攻守のバランス感覚を養う。
- **Meta-Game League (メタゲーム・リーグ)**: 過去の自分や固定デッキを含む「仮想リーグ」を構築し、常に多様な敵と戦わせることで、メタ（流行）の循環に対応する。

## 2. Phase 1: Dual Curriculum (二極化カリキュラム)
AIに「攻め」と「守り」の極意を別々に、しかし同じ脳（ニューラルネットワーク）で学習させるフェーズ。

### 2.1 コンセプト
- **Aggro Mode (速攻脳)**: 相手を倒す「速度」を学習する。
- **Control Mode (耐久脳)**: 相手のリソースを枯らす「盤面支配」を学習する。
- **統合**: これらを交互に行うことで、AIは「今は攻めるべきか、守るべきか」という大局観 (Value Network) を獲得する。

### 2.2 実装ロジック
学習ループ内で、エピソードごとに使用デッキと報酬体系を切り替える。

#### 学習ループ (疑似コード)
```python
for episode in range(total_episodes):
    # 偶数回はアグロ、奇数回はコントロールを使用
    if episode % 2 == 0:
        mode = "AGGRO"
        my_deck = AGGRO_DECK_LIST
        # 報酬: 短期決戦ボーナスあり
        reward_config = {"win": 1.0, "turn_penalty": 0.05, "shield_break": 0.1}
    else:
        mode = "CONTROL"
        my_deck = CONTROL_DECK_LIST
        # 報酬: リソース差ボーナスあり、長期戦許容
        reward_config = {"win": 1.0, "turn_penalty": 0.0, "enemy_hand_discard": 0.05}

    # 環境のリセット
    state = env.reset(my_deck=my_deck, reward_config=reward_config)

    # ... (対戦処理)
```

### 2.3 期待される効果
- **アグロ学習時**: マナを使い切り、リーサル（決着）を見逃さない計算力を習得。
- **コントロール学習時**: シールドトリガーを計算に入れ、ハンドアドバンテージ（手札差）を広げる立ち回りを習得。
- **結果**: 「コントロールデッキを使っているのに、相手が事故っていると見るやアグロのように攻める」といった柔軟な判断が可能になる。

## 3. Phase 2: Meta-Game League (メタゲーム・リーグ)
Phase 1で基礎体力をつけたAIを、より過酷な「異種格闘技戦」に投入するフェーズ。

### 3.1 リーグの構成 (League Participants)
対戦相手（Opponent）を単一のAIにせず、以下のプールからランダムに選出する。

| ID | タイプ | 説明 | 役割 |
|---|---|---|---|
| **Main** | Learning | 現在学習中の最新AI | 主人公 |
| **Past_V1** | Frozen | 1000エピソード前の自分 | 退化の防止（過去の自分に負けないこと） |
| **Meta_A** | Fixed | 環境トップのアグロデッキ (固定ロジック) | 最速の動きに対する防御テスト |
| **Meta_B** | Fixed | 環境トップの除去コン (固定ロジック) | 妨害に対するリカバリーテスト |
| **Combo_X** | Specialist | 特定のコンボ特化AI | 初見殺し・ソリティアへの対応 |

### 3.2 マッチメイキング・システム
ランダムに対戦相手を選ぶが、「勝てていない相手」との対戦確率を上げる（優先的サンプリング）。

```python
class LeagueManager:
    def __init__(self):
        self.opponents = ["Past_V1", "Meta_A", "Meta_B", "Combo_X"]
        self.win_rates = {name: 0.5 for name in self.opponents} # 初期勝率
    
    def get_opponent(self):
        # 勝率が低い相手ほど選ばれやすくする (苦手克服)
        weights = [1.0 - self.win_rates[name] for name in self.opponents]
        return random.choices(self.opponents, weights=weights, k=1)[0]
    
    def update_result(self, opponent_name, is_win):
        # 勝率の更新 (移動平均)
        current = self.win_rates[opponent_name]
        self.win_rates[opponent_name] = current * 0.9 + (1.0 if is_win else 0.0) * 0.1
```

### 3.3 リーグの更新 (League Update)
Main Agentが強くなりすぎた場合、練習相手も更新する。

- **殿堂入り判定**: 特定の固定デッキ（例: Meta_A）に対する勝率が 80% を超えた。
- **更新**:
    - その時点の Main Agent をコピーし、「Past_V2」 としてリーグに追加する。
    - または、AIが生成した「新しい最強デッキ」を装備した固定AIを新規参入させる。
- **削除**: リーグ内のAIが増えすぎたら、最も弱い（全員に負け越している）AIを削除する。

## 4. 統合ワークフロー
Kaggle等のPC電源OFF環境における実行フロー。

1.  **初期化**:
    - Datasetから「Main Model」と「League Info (対戦相手リスト)」をロード。
2.  **カリキュラム選択**:
    - 現在の学習進度に応じてデッキを選択 (Aggro or Control)。
3.  **対戦相手決定**:
    - LeagueManager が対戦相手（過去モデル or 固定デッキ）を決定。
4.  **対戦 & 学習 (PPO)**:
    - 指定された「自分のデッキ」vs「相手のデッキ」で高速対戦。
5.  **評価 & 更新**:
    - 1時間の学習ごとにリーグの勝率を確認。
    - 圧倒しているならリーグメンバーを入替・更新。
6.  **保存**:
    - Main Modelと、更新されたLeague InfoをDatasetに保存。

## 5. 結論
このシステムにより、AIは以下のサイクルを繰り返して無限に強くなる。

- **二刀流**: 攻めと守りの両方のプレイングをマスターする。
- **弱点克服**: 苦手なデッキタイプと集中的に戦わされる。
- **世代交代**: 過去の自分を乗り越え、常に「今の自分」にとって最も手強い相手と練習し続ける。
