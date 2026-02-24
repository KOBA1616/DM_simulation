# C++エンジン関連コードベース調査報告書

本報告書は、現在のC++エンジン関連のディレクトリ構成、重複実装、ActionからCommand方式への移行状況、および削除候補のレガシーコードに関する網羅的な調査結果をまとめたものです。

## 1. ディレクトリ構成と役割

現在のプロジェクト構造は、以下の主要なディレクトリに分かれています。

### `src/engine` (C++ コアエンジン)
ゲームロジックの中核となる実装が含まれています。
- **`systems/`**: ゲームのルール、効果、フェーズ進行などを管理する各システム (`flow`, `rules`, `mechanics`, `effects` 等)。
- **`core/`**: 基本的なデータ構造 (`Card`, `Player`, `GameState` 等)。
- **`infrastructure/`**: データ読み込み (`JsonLoader`) や依存関係管理。
- **`utils/`**: ユーティリティ関数。

### `src/bindings` (Python バインディング)
C++エンジンをPythonから利用するための pybind11 バインディング実装です。
- **`bind_engine.cpp`**: `PhaseManager` や `CommandSystem` などの主要コンポーネントを公開。
- **`bind_core.cpp`**: `GameState` や `Player` などのデータ構造を公開。

### `dm_toolkit` (Python ツールキット)
エンジンのPython側ラッパー、ユーティリティ、および移行用レイヤーです。
- **`dm_ai_module.py`**: `dm_ai_module` (ネイティブ拡張) へのプロキシ。
- **`action_to_command.py`**: レガシーなAction辞書を新しいCommand辞書に変換する重要な移行レイヤー。
- **`commands_v2.py`**: 新しいCommandベースの生成ロジック。
- **`unified_execution.py`**: 統合実行パイプライン（Action/Command双方に対応）。

### `scripts` (ユーティリティスクリプト)
開発、テスト、ビルド用のスクリプト群。
- 多くのファイルが含まれていますが、`generate_card_tests.py` や `migrate_model.py` など、一部は古い構成を前提としている可能性があります。

---

## 2. 重複実装 (Duplicate Implementations)

最も顕著な重複実装は、**PythonによるC++エンジンの模倣 (Shim)** です。

### `dm_ai_module.py` (ルートディレクトリ)
このファイルは、C++拡張モジュール (`dm_ai_module.so` / `.pyd`) が存在しない環境でも動作するように、Pythonでエンジンの主要クラス (`GameInstance`, `CommandSystem`, `PhaseManager` 等) を再実装しています。
- **問題点**: C++側のロジック変更が自動的に反映されないため、二重管理のコストが発生しています（例: マナチャージのロジックやフェーズ進行など）。
- **現状**: テスト環境や軽量なスクリプト実行のために維持されていますが、`IS_NATIVE = False` の場合の挙動はC++版と完全には一致しません。

### `dm_toolkit/dm_ai_module.py`
ルートディレクトリの `dm_ai_module` をインポートし、その内容を自身の名前空間にコピーするプロキシモジュールです。
- **重複**: 実質的なロジックはなく、単なるエイリアスとして機能しています。プロジェクト全体で `import dm_ai_module` (ルート) に統一すれば不要となる可能性があります。

---

## 3. ActionからCommand方式への移行状況

移行は **「ハイブリッド（過渡期）」** の状態にあります。完全なネイティブ化（Command方式）には至っていませんが、主要なパイプラインはCommandベースに移行しつつあります。

### 現状のアーキテクチャ
1.  **`dm_toolkit/action_to_command.py`**:
    -   これが現在の移行の要です。レガシーな「Action辞書」（`type`, `value1` 等の古い形式）を、新しい標準化された「Command辞書」に変換します。
    -   `map_action` 関数がすべての変換を一手に引き受けており、この層がある限り、古いUIコードやAIモデルは変更なしで動作します。

2.  **`dm_toolkit/commands_v2.py` vs `commands.py`**:
    -   **`commands_v2.py`**: ネイティブの `generate_commands` (C++) を優先して呼び出す新しい実装です。
    -   **`commands.py`**: 古い `ActionGenerator` やPython側のロジックを使用するレガシー実装です。
    -   `v2` は、ネイティブ実装が利用できない場合や失敗した場合に、依然として `legacy.generate_legal_commands` (`commands.py`) にフォールバックする仕組みを持っています。

3.  **統合実行 (`unified_execution.py`)**:
    -   入力を受け取り、それがActionであれば `action_to_command.py` を通してCommandに変換し、最終的に `EngineCompat` (C++) で実行するフローを確立しています。

### 結論
「完全移行」とは、**`action_to_command.py` が不要になり、すべてのシステムが最初からCommand辞書（または構造体）を生成・消費する状態** を指します。現在はまだ変換レイヤーに強く依存しています。

---

## 4. 削除候補のレガシーコード

以下のファイルやモジュールは、移行完了後に削除または整理すべき候補です。

### 優先度: 高 (整理推奨)
1.  **`dm_toolkit/dm_ai_module.py`**:
    -   役割が薄く、ルートの `dm_ai_module` への直接インポートに統一することで削除可能です。

### 優先度: 中 (移行完了待ち)
1.  **`dm_toolkit/commands.py`**:
    -   `commands_v2.py` のフォールバック先として機能しているため、現時点では削除できません。C++側の `generate_commands` が完全にすべてのケースをカバーし、Python側のフォールバックが不要になった時点で削除可能です。

2.  **`dm_toolkit/unified_execution.py` (v1)**:
    -   `unified_execution_v2.py` が存在しますが、v1は複雑な変換ロジック (`to_command_dict`, `ensure_executable_command`) を持っています。これらが不要になる（全入力が既に正しいCommand形式になる）までは維持が必要です。

### 優先度: 低 (Python Shim)
1.  **`dm_ai_module.py` (ルートのPython実装部分)**:
    -   開発体験（ビルドなしでの動作確認）のために便利ですが、厳密には「レガシー」かつ「技術的負債」です。長期的には削除し、C++拡張の利用を必須とするか、インターフェース定義のみを残す形が望ましいです。
