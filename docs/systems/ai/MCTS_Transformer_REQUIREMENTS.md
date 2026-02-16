# MCTS + Transformer 統合 要件定義書

作成日: 2026-01-27
作成者: 自動生成（GitHub Copilot 補助）

目的
- 既存 C++ エンジン上で MCTS と Transformer（注意機構ベースのポリシー/バリューネット）を組み合わせ、効率的で高性能な探索と学習パイプラインを構築する。

範囲
- C++ エンジンの探索（MCTS）コアを使用し、Transformer による事前確率（policy logits）と状態価値（value）を評価器として組み込む。
- バッチ推論は低遅延で実行可能にする。 ONNX Runtime または LibTorch を用いる。
- Python 側の学習スクリプトおよび ONNX エクスポートパイプラインと連携する。

現状の実装サマリ（重要なコンポーネント）
- MCTS コア（C++）
  - ファイル: `src/ai/mcts/mcts.cpp`, `src/ai/mcts/mcts.hpp`
  - 機能: PUCT ベース選択、展開、バックプロパゲーション、バッチ評価呼び出し、PIMC（部分情報）処理、探索ノイズ付与。
  - evaluator として C++ の callback を受け取り、バッチ単位でモデルを評価する設計。

- ゲームエンジン（C++）
  - 主な API: `GameState::clone()`, `PhaseManager::fast_forward()`, `IntentGenerator::generate_legal_actions()`, `GameLogicSystem::resolve_action()`。
  - これらは MCTS のシミュレーションでそのまま使用可能（再実装不要）。

- エンコーディング／アクション変換（C++）
  - `ai/encoders/TokenConverter`：状態→トークン系列
  - `ai/encoders/TensorConverter`：状態→フラットテンソル
  - `ai/encoders/CommandEncoder`：Command → 固定長インデックス（`TOTAL_COMMAND_SIZE`）

- ニューラル評価器（C++）
  - `src/ai/evaluator/neural_evaluator.cpp`（`NeuralEvaluator`）
  - ONNX Runtime 経由または Python コールバック経由でバッチ推論を行える実装を持つ。
  - `NeuralEvaluator::load_model()` により ONNX モデルを読み込める（ビルド時に `USE_ONNXRUNTIME` が必要）。

- Python 側学習／エクスポート
  - トレーニングモデル: `dm_toolkit/ai/agent/transformer_model.py` (`DuelTransformer`) — PyTorch 実装済み。
  - 訓練スクリプト: `python/training/train_transformer_phase4.py`。
  - ONNX エクスポート: `training/export_model_to_onnx.py`（.pth → .onnx）。
  - デプロイ確認スクリプト: `training/deploy_with_onnx.py`（C++ の `NeuralEvaluator` に ONNX を読み込ませて `ParallelRunner.play_games` を実行）。

- バッチ推論ブリッジ／Python コールバック
  - `bindings/python_batch_inference.cpp` と `bind_ai.cpp` により、C++ から Python のバッチ評価コールバックを登録・呼び出せる。
  - `NeuralEvaluator` は `dm::python::call_sequence_batch_callback` などを利用して Python 推論を呼べる。

- 推論ランタイムオプション（CMake）
  - `CMakeLists.txt` に `option(USE_ONNXRUNTIME ON)`（デフォルト：ONにしている箇所あり）と `option(USE_LIBTORCH OFF)` があり、ビルド時に有効化可能。
  - ONNX Runtime を FetchContent で取得・リンクする仕組みがある（Windows/Linux/macOS 向けのダウンロード URL を定義）。

設計要件（機能要件）
1. MCTS 側
   - MCTS は外部 evaluator をバッチで呼び出す API を使用する。
   - 展開時に `IntentGenerator::generate_legal_commands()` を呼び出し、`CommandEncoder::command_to_index()` で policy のインデックスへマップする。
   - ルート展開時に事前確率にディリクレノイズを付与する。

2. Transformer 側（推論）
   - モデルは入力としてトークン列（`TokenConverter::encode_state`）を受け取り、出力として `(policy_logits:[action_dim], value:[1])` を返す。出力 action_dim は `CommandEncoder::TOTAL_COMMAND_SIZE` と整合すること。
   - 推論はバッチ処理対応であり、複数ノードの状態をまとめて評価できること。
   - 低遅延のため ONNX Runtime（`USE_ONNXRUNTIME`）あるいは LibTorch（`USE_LIBTORCH`）のどちらかを用いる。ONNX の場合は `training/export_model_to_onnx.py` により生成したファイルを利用。

3. 学習パイプライン
   - 自己対戦データを `ParallelRunner` を用いて収集し、(state_tokens, π_target, z) を保存する。
   - 学習ロジックは既存の `python/training/train_transformer_phase4.py` を使用可能。
   - 学習後、`export_model_to_onnx.py` で ONNX へ変換すること。

非機能要件
- レイテンシ: 単一評価バッチあたりのレイテンシを最小化する（目標: 1ms 台〜数十ms、環境に依存）。
- スケーラビリティ: 複数スレッド/プロセスでの並列探索をサポートする（`ParallelRunner` あり）。
- 再現性: 学習と評価で同じ `TokenConverter` と `CommandEncoder` を使用し、シード管理を行う。

現状のギャップ（要対応）
- ビルド環境: C++ 側で `USE_ONNXRUNTIME` を有効にするには環境でのダウンロード（FetchContent）・リンクが必要。Windows では事前に Visual Studio と適切なツールチェインが必要。
- 入出力整合性: `TokenConverter` のトークン仕様（語彙サイズ・パディング ID 等）と `DuelTransformer` の `vocab_size`、および `CommandEncoder::TOTAL_COMMAND_SIZE` が一致していることを明示的に検証する必要あり。
- 最適化: 実運用では推論のバッチ化・キャッシュ・非同期化・スレッドプールなどの最適化が必要。

検証項目（スモークテスト）
1. 環境準備: `cmake -S . -B build -DUSE_ONNXRUNTIME=ON` が成功すること（FetchContent による ONNX Runtime ダウンロードを含む）。
2. ビルド: `cmake --build build --config Release` が成功して `bin/dm_ai_module`（Python 拡張）が生成されること。
3. デプロイテスト: `python training/deploy_with_onnx.py --onnx models/<model>.onnx` を実行して `NeuralEvaluator.load_model()` と `ParallelRunner.play_games()` が正常に動作すること（win_rate の出力）。
4. 整合テスト: MCTS を走らせ、`CommandEncoder` インデックスへのマッピングが正しく行われ、生成された policy が違法行動を受け取らないことを確認する。

移行手順（簡潔）
1. 開発環境で C++ ビルドを ONNX 有効化で行う（CMake オプションを指定）。
2. 既存の学習スクリプトでモデルを学習し、`training/export_model_to_onnx.py` で ONNX に変換。
3. `training/deploy_with_onnx.py` で ONNX を読み込み、C++ 側の `NeuralEvaluator` を経由して `ParallelRunner` を使った自己対戦評価を行う。
4. 期待されるパフォーマンスを達成するためにバッチサイズ・探索回数・非同期化をチューニングする。

保守・運用上の注意
- ONNX / LibTorch のバージョンにより互換性が変わるため、`export_model_to_onnx.py` の opset を固定または検証パスを用意すること。
- トークン／アクション仕様を変更する場合は古いデータとの互換性を保つエクスポート/変換ツールを用意すること。
- セキュリティ: 外部から受け取るモデルのロード時は信頼できる場所のファイルのみを許可すること。

付録: 主要ファイル一覧（抜粋）
- C++ MCTS: `src/ai/mcts/mcts.cpp`, `src/ai/mcts/mcts.hpp`
- NeuralEvaluator / ONNX: `src/ai/evaluator/neural_evaluator.cpp`, `src/ai/inference/onnx_model.*`
- Token/Tensor/Action Encoders: `src/ai/encoders/token_converter.cpp`, `src/ai/encoders/tensor_converter.cpp`, `src/ai/encoders/command_encoder.cpp`
- Python Transformer: `dm_toolkit/ai/agent/transformer_model.py`
- Training: `python/training/train_transformer_phase4.py`, `training/export_model_to_onnx.py`
- Deploy/Test: `training/deploy_with_onnx.py`, `src/bindings/python_batch_inference.cpp`, `src/bindings/bind_ai.cpp`

---

ドキュメント終わり。
