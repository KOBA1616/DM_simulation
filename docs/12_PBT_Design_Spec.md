# PBT (Population-Based Training) 導入設計書
## ～ "PC電源OFF" 環境における自律進化型AI集団の構築 ～

## 1. 概要
本ドキュメントは、単一のAIモデルではなく、**「AIの集団（Population）」**を育成し、互いに競わせることでメタゲーム（戦略の流行廃り）を解決し、最強のデッキとプレイングを自動獲得するシステム「PBT」の設計仕様である。

### 目的
*   **メタゲームの自動解決**: 「速攻」「コントロール」「コンボ」など、異なる戦略を持つAIを共存・進化させる。
*   **パラメータ調整の自動化**: 学習率や報酬設計などのハイパーパラメータを、AI自身に探索させる。
*   **Kaggle環境への適応**: 9時間の制限時間内で効率的に「集団学習」を行う軽量化設計。

## 2. システムアーキテクチャ

### 2.1 集団の構成 (The Population)
リソース制約（Kaggle GPU x1）を考慮し、**「少数精鋭の4エージェント」**で構成する。

| Agent ID | 役割・初期設定（例） | 特徴 |
| :--- | :--- | :--- |
| **Agent_A** | Standard | 標準的なパラメータ設定。安定志向。 |
| **Agent_B** | Aggressive | 攻撃報酬高め。アグロデッキを好みやすい。 |
| **Agent_C** | Defensive | 防御・生存報酬高め。コントロールデッキを好みやすい。 |
| **Agent_D** | Explorer | 学習率高め・ランダム性高め。奇抜なデッキ開発担当。 |

### 2.2 データ構造 (Dataset)
Kaggle Dataset内にフォルダ `pbt_population` を作成し、以下の構造で永続化する。

```text
pbt_population/
├── generation_info.json    # 現在の世代数、対戦成績履歴
├── agent_A/
│   ├── model.pth           # 脳（ニューラルネット）
│   ├── deck.json           # 現在使用しているデッキ
│   └── config.json         # 現在のハイパーパラメータ（学習率など）
├── agent_B/ ...
├── agent_C/ ...
└── agent_D/ ...
```

## 3. 運用サイクル (The Cycle)
Kaggle Notebookが起動するたびに、以下の "Evolution Loop" を1回実行する。

### Step 1: 順次学習 (Sequential Training)
GPUが1つしかないため、時間を分割して順番に学習させる。
*   **配分**: 9時間 ÷ 4体 = 1体あたり約2時間
*   **処理**:
    *   Datasetから `Agent_X` をロード。
    *   2時間、自己対戦（または過去の自分との対戦）を行い、強化学習 (PPO) を回す。
    *   同時に「デッキ構築（山登り法）」も行い、デッキを最適化する。
    *   結果を一時保存。

### Step 2: リーグ戦 (Evaluation)
学習後の4体を総当たりさせ、実力を測定する。
*   **形式**: 全員 vs 全員 (各100戦程度)
*   **指標**: 「勝率」および「デッキの多様性スコア（他と違うか）」。

### Step 3: 搾取と探索 (Exploit & Explore)
PBTの核心部分。成績下位のモデルを淘汰し、上位モデルをベースに進化させる。
*   **ランキング**: 勝率順に並べる（例: 1位=A, 4位=D）。
*   **淘汰 (Exploit)**: 最下位(D)を削除する。
*   **継承 (Clone)**: 1位(A)の「モデル」と「パラメータ」をコピーし、Dの枠に入れる。
    *   **注意**: デッキはコピーせず、Dが持っていたものを維持（または少しランダム変更）して、多様性を保つ。
*   **変異 (Explore)**: 新生Dのハイパーパラメータをランダムに少し変更する。
    *   例: `learning_rate` を 1.2倍 または 0.8倍 にする。
    *   例: 「マナ加速の報酬」を少し上げる。

### Step 4: 保存 (Persistence)
次回のNotebook起動用に、更新された4体をDatasetに上書き保存する。

## 4. 詳細ロジック：多様性の保護 (Niche Protection)
4体全員が「最強の速攻デッキ」になってしまうと、進化が止まる（メタが回らなくなる）。これを防ぐため、**「他と違うデッキを使っているAI」**を保護する。

### 実装: デッキ類似度チェック
淘汰フェーズ（Step 3）において、単純な勝率だけでなく「独自性」を評価に加える。

```python
def calculate_diversity_score(target_deck, other_decks):
    # 他の3人のデッキとの「カード一致率」の平均を出す
    similarity = mean([jaccard_similarity(target_deck, d) for d in other_decks])
    # 似ていないほどスコアが高い
    return 1.0 - similarity

# 最終スコア = 勝率 + (多様性重み * 多様性スコア)
final_score = win_rate + (0.2 * diversity_score)
```

これにより、**「勝率は2位だが、唯一コントロールデッキを使っているAgent_C」**が生き残りやすくなる。

## 5. Kaggle 実装コード構成案

```python
# PBT_Manager.py (疑似コード)

class PBTManager:
    def __init__(self, population_size=4):
        self.agents = self.load_agents_from_dataset()
    
    def run_cycle(self, total_hours=9):
        # 1. 順次学習
        train_time_per_agent = (total_hours - 1) / 4 # 評価に1時間残す
        
        for agent in self.agents:
            print(f"Training {agent.name} for {train_time_per_agent} hours...")
            agent.train(duration=train_time_per_agent)
            
        # 2. リーグ戦
        results = self.run_tournament()
        print("League Results:", results)
        
        # 3. 進化 (Exploit & Explore)
        self.evolve(results)
        
        # 4. 保存
        self.save_all()

    def evolve(self, results):
        # 勝率でソート
        sorted_agents = sorted(self.agents, key=lambda x: x.score, reverse=True)
        best_agent = sorted_agents[0]
        worst_agent = sorted_agents[-1]
        
        # 下位25%を淘汰し、上位のコピーで置換
        print(f"Replacing {worst_agent.name} with mutated {best_agent.name}...")
        
        # Copy Weights
        worst_agent.model.load_state_dict(best_agent.model.state_dict())
        
        # Mutate Hyperparams (学習率などを±20%変動)
        worst_agent.mutate_hyperparams()
        
        # Reset Optimizer (勢いをリセット)
        worst_agent.reset_optimizer()
```

## 6. メリット・デメリット分析

### メリット (Pros)
*   **メタゲームの再現**:
    *   Agent_A(速攻)が流行る → Agent_B(除去)が進化してAを倒す → Agent_C(コンボ)がBの隙を突く……という「環境の循環」が自動発生する。
*   **パラメータ調整からの解放**:
    *   「学習率は 1e-4 がいいか 3e-4 がいいか？」と悩む必要がない。良い設定を持つ個体が勝手に生き残る。
*   **堅牢性 (Robustness)**:
    *   特定のハメ技に弱い「穴のあるAI」が生まれにくい（集団の誰かに咎められるため）。

### デメリット (Cons)
*   **進行速度**:
    *   1体あたりの学習時間が短くなるため、個々のプレイング精度向上は単独学習より遅い。
    *   **対策**: デッキ構築が落ち着いたら、PBTを停止して、最強の1体を長時間学習させるフェーズに移行する。
*   **メモリ管理**:
    *   Pythonスクリプト内でモデルのロード/アンロードを適切に行わないと、GPUメモリが溢れる。
    *   **対策**: 1体の学習が終わるごとに `del model`, `torch.cuda.empty_cache()` を確実に実行する。

## 7. 結論
PBTは、PC電源OFFでの長期運用において**「放置していても勝手に賢くなる」ための最高のフレームワークである。単なる強化学習を超えて、「AIたちが切磋琢磨するコロシアム」**をKaggle上に構築することで、未知の最強デッキが産まれる可能性を最大化できる。
