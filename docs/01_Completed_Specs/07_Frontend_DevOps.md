# 7. GUI & 開発ツール (Frontend/DevOps)

## 7.1 PyQt6 GUI [Updated]
- **Control**: クリックによるステップ実行.
- **Visualization**:
    - **MCTS Graph View**: 探索木をグラフィカルに表示し、AIの思考プロセスを可視化 [New].
    - **Card Detail Panel**: ホバー時にカードの詳細スペックを表示 [New].
    - **God View**: デバッグ用に相手の手札・シールドを透視するモード [New].
- **Concurrency**: ポーリング方式による非同期更新.

## 7.2 開発補助
- **Card Generator**: `tools/card_gen/`
    - JSON定義ファイル (`data/card_effects.json`) からC++のカード効果実装コード (`generated_effects.hpp`) を自動生成。
    - **Supported Effects**: `mana_charge`, `draw_card`, `tap_all`, `destroy`, `bounce`.
    - 単純な効果（ドロー、マナブースト、破壊、バウンス）の実装工数を大幅に削減。
- **Deck Builder**: GUI内蔵エディタ.
