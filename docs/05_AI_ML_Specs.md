# 5. AI & 機械学習仕様 (AI/ML Specs)

## 5.1 ネットワーク構造
- **Backbone**: MLP (多層パーセプトロン) 5層 + Embedding Layers.
- **Architecture**: **MLP (Multi-Layer Perceptron)**
    - **Decision**: Set Transformer等の複雑なアーキテクチャは学習コストと実装難易度が高いため採用せず、**高速かつ安定したMLP**をベースに開発を進める。
    - **Focus**: ネットワーク構造の複雑化よりも、**入力特徴量の質（LLM Embeddings等）**と**学習手法（League Training）**の改善にリソースを集中する。
- **Input Strategy (Embeddings)**:
    - **Spec Embeddings**: カードID（One-hot）の代わりに、**カードスペック（コスト、パワー、文明、種族、効果フラグ）** を入力特徴量として使用。
    - **LLM Text Embeddings (NLP × RL)**:
        - **Method**: BERTやSBERT等の軽量LLMを用いて、カードのテキスト（効果文）を事前にベクトル化（例: 768次元）する。
        - **Dimensionality Reduction (次元圧縮)**:
            - 生の768次元はゲーム状態（盤面情報）に対して過大であり、学習のバランスを崩すリスクがある。
            - **Projection Layer**: 学習可能な線形層（Linear Layer）を挟み、**768次元 → 128次元（ハイパーパラメータ）** 程度に圧縮してからメインネットワークに入力する。
            - **Design Rationale**: カードスペック情報（コスト、パワー等）の埋め込みベクトル（約32〜64次元想定）とバランスを取り、Set Transformerの入力次元（$d_{model}$）に合わせる設計とする。
        - **Zero-shot Adaptation**: 「引く」という単語のベクトル類似性を通じて、未学習のカードでも「ドローソースである」と認識可能にする。
        - **Efficiency**: ベクトル化は事前計算（Pre-computation）するため、学習・推論時のオーバーヘッドは最小限。
    - これにより、未知のカードへの汎化性能を高め、モデルサイズを削減する。
- **Flow**: Input (Specs) -> Embedding -> [Set Transformer Block] -> Pooling -> FC(1024) -> ReLU -> ... -> FC(ActionSize).
- **Optimization**:
    - **Mixed Precision (AMP)**: 学習時に float16 を使用し、メモリ帯域を節約。
    - **Quantization**: 推論時に int8 量子化を行い、CPU推論速度を2〜3倍に高速化。

## 5.2 入出力マッピング
- **Input Tensor**: Zero-Copy (C++ -> Python/LibTorch).
- **Action Space**: Flattened Fixed Vector (approx. 600 dim).

## 5.3 MCTS (Monte Carlo Tree Search) [Updated]
- **Implementation**: C++ (`src/ai/mcts/`) による高速実装。
- **Algorithm**: **AlphaZero Style**
    - 探索結果（訪問回数分布）を教師データとしてPolicy Networkを学習させる「教師あり学習」アプローチを採用。
- **Evaluator**:
    - **Heuristic**: ルールベースの高速評価関数 (`HeuristicEvaluator`). GUI/Debug用。
    - **Neural**: LibTorch (C++) モデルによる推論。
- **Determinization**: 未公開情報をランダム固定してプレイアウト (C++側で処理).
- **Performance**: Python版と比較して約100倍の探索速度を実現。

## 5.4 学習ループ
- **Replay Buffer**: Hybrid Buffer (Sliding Window + Golden Games).
- **Seed Mgmt**: 学習マネージャーによるシード配布.

## 5.5 リーグ学習 (League Training)
- **Objective**: 自己対戦（Self-Play）における「忘却（Catastrophic Forgetting）」と「ジャンケンの円環（Rock-Paper-Scissors Cycle）」の防止。
- **Method**:
    - 過去のチェックポイント（モデル）を「リーグ（対戦相手プール）」として保存。
    - 学習中のエージェント（Main Agent）の対戦相手を以下の比率でランダムに選択する。
        - **80%**: 最新の自分（Self-Play） - 強くなるため。
        - **10%**: 過去の自分（Past Agents） - 弱点克服と忘却防止のため。
        - **10%**: ルールベースBot（Rule-based Bots） - 特定のハメ技や極端な戦略への対策。
- **Effect**: 特定のメタに過剰適応することを防ぎ、あらゆる戦法に対応できるロバスト（堅牢）なAIを育成する。

## 5.6 Population Based Training (PBT)
- **詳細仕様**: [12. PBT導入設計書 (PBT Design Spec)](./12_PBT_Design_Spec.md) を参照。
- **Objective**: ハイパーパラメータ（学習率、割引率、MCTS探索パラメータ等）の自動最適化とメタゲームの解決。
- **Method**:
    - **Parallel Workers**: Kaggle環境等のリソース制約下で、4体のエージェントを順次学習・評価する。
    - **Exploit & Explore**:
        - 定期的（1サイクルごと）に各ワーカーの性能（勝率）を評価。
        - 下位のワーカーを停止し、上位ワーカーのモデルパラメータとハイパーパラメータをコピー（Exploit）。
        - コピーしたハイパーパラメータにランダムな変異（Mutation）を加えて再開（Explore）。
    - **Niche Protection**: デッキの多様性を評価に加え、単一戦略への収束を防ぐ。
- **Effect**: 手動によるグリッドサーチを不要にし、学習の進行に合わせて動的に最適なハイパーパラメータを発見する。限られた計算リソースを有効活用する。

## 5.7 Auxiliary Tasks (補助タスク学習)
- **Objective**: 「勝敗」以外の情報も吸い尽くし、スパースな報酬問題を解決する。
- **Problem**: 強化学習では数百ターンプレイして最後に「勝った/負けた」という情報しか得られないため、途中のどの手が良かったのか理解するのに時間がかかる。
- **Method**: メインの「勝ち方」を学習するネットワークの途中に分岐を作り、以下の**サブクエスト（予測問題）**も同時に解かせる。
    - **Next State Prediction**: 「このカードを出したら、自分のマナと手札はどう変化するか？」
    - **Inverse Dynamics Prediction**: 「今の状態から次の状態になったということは、どのアクションを選んだのか？」
    - **Hidden Information Prediction**: 「相手のマナや墓地の状況から推測して、相手の手札に『スパーク』がある確率は？」
- **Effect**: AIは「勝つため」だけでなく「ルールと盤面の因果関係」を理解せざるを得なくなる。結果として、特徴抽出層（CNNやTransformer部分）の学習が爆速になり、少ないエピソード数で賢くなる（DeepMindのUnrealエージェント等の手法）。

## 5.8 Meta-Game Solver (PSRO)
- **Objective**: 「井の中の蛙」を防ぎ、ナッシュ均衡点（弱点のない完全な戦略）に近づく。
- **Problem**: 単純な自己対戦では、特定の戦略には強いが未知の戦略に弱いという偏りが生じたり、三すくみの循環に陥る可能性がある。
- **Method**: **PSRO (Policy Space Response Oracles)**
    - **Meta-Game Analysis**: リーグ内の全エージェント間の勝率行列（Payoff Matrix）を作成し、メタゲームの均衡解（メタ戦略）を計算する。
    - **Adaptive Sampling**:
        - ランダムに対戦相手を選ぶのではなく、**「現在のメタゲームにおいて、学習中のエージェントにとって最も脅威となる（攻略すべき）相手」**を優先的にサンプリングする。
        - 例: 三すくみ（A > B > C > A）において、自分がAなら、苦手なCの出現率を上げる。
- **Effect**: 常に「今の自分にとって一番痛いところを突いてくる相手」と戦う環境を自動構築し、特定のメタに依存しない真の強さを獲得する。
