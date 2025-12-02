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

## 5.5 Meta-Game Curriculum & League (メタゲーム・カリキュラム)
- **詳細仕様**: [14. メタゲーム・カリキュラム設計書 (Meta-Game Curriculum Spec)](./14_Meta_Game_Curriculum_Spec.md) を参照。
- **Objective**: 特定のデッキにしか勝てない「特化型AI」ではなく、あらゆる戦況に対応できる「汎用型AI」を育成する。
- **Phase 1: Dual Curriculum (二極化カリキュラム)**:
    - **Aggro Mode**: 速度とリーサル計算を学習。
    - **Control Mode**: リソース管理と盤面支配を学習。
    - これらを交互に行うことで、攻守のバランス感覚（Value Network）を養う。
- **Phase 2: Meta-Game League (メタゲーム・リーグ)**:
    - **Participants**: 最新の自分、過去の自分、環境トップの固定デッキ（アグロ・コントロール）、コンボ特化AI。
    - **Adaptive Matchmaking**: 勝率の低い（苦手な）相手と優先的に対戦させることで、弱点を効率的に克服する。
- **Effect**: 「ジャンケンの円環」を抜け出し、メタゲームの進化に合わせて成長し続けるAIを実現する。

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

## 5.9 Hidden Information Inference (POMDP)
- **詳細仕様**: [13. 非公開領域推論システム設計書 (POMDP Inference Spec)](./13_POMDP_Inference_Spec.md) を参照。
- **Objective**: 「見えていない情報（相手の手札、シールド）」を確率的に推論し、リスク管理とブラフ読みを可能にする。
- **Method**:
    - **Self-Inference (O(1))**: 自分の山札・シールドの中身は、初期デッキと公開情報の差分から正確な確率分布（16次元スタッツ）を計算して入力する。
    - **Opponent-Inference (Distillation)**:
        - **Teacher Model**: 学習時のみ、相手の手札が見えている「神の視点」を持つ教師モデルを使用。
        - **Student Model**: 本番用モデルは、教師モデルの行動分布（Logits）を模倣するように学習（蒸留）することで、間接的に「相手の手札を読む直感」を獲得する。
- **Effect**: 「山札にトリガーがないからブロックする」「相手がマナを残したから警戒する」といった、上級者のような高度な判断を実現する。

## 5.10 Result Stats & Autonomous Evolution (結果スタッツと自律進化)
- **詳細仕様**: [15. 結果スタッツ設計書 (Result Stats Spec)](./15_Result_Stats_Spec.md) を参照。
- **Objective**: 人間の主観（タグ付け）を排除し、シミュレーション結果のみから「カードの強さ」を定義する。
- **Method**:
    - **16-dim Stats**: 各カードの使用タイミング、アドバンテージ、勝率貢献度などを16次元のベクトルとして数値化。
    - **Learning Cycle**:
        1.  **Observer**: ランダム対戦で初期スタッツを収集。
        2.  **Learner**: スタッツを元にAIがプレイングとデッキ構築を学習。
        3.  **Refinement**: 賢くなったAI同士の対戦でスタッツを更新（コンボカードの再評価）。
- **Effect**: 未知のカードや新弾カードに対しても、AIが自ら「使い方」と「強さ」を発見し、メタゲームを進化させ続ける。

## 5.11 Scenario Training (シナリオ・トレーニング)
- **詳細仕様**: [16. シナリオ・トレーニング設計書 (Scenario Training Spec)](./16_Scenario_Training_Spec.md) を参照。
- **Objective**: 確率の低い複雑なコンボや、終盤の詰み手順（リーサル）を効率的に学習させる。
- **Method**:
    - **Scenario Setup**: C++エンジンに「特定の盤面（手札、マナ、場）」からゲームを開始する機能を実装。
    - **The Drill**: 学習初期に「コンボパーツが揃った状態」からの特訓を繰り返すことで、手順を丸暗記させる。
    - **Loop Detection**: 同一局面の繰り返しを検知し、「ループ証明完了」として報酬を与える。
- **Effect**: ランダム探索では発見困難な無限ループや即死コンボを、AIが確実に実行できるようになる。
