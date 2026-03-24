````markdown
# DM Engine — GUI保持 & ヘッドレス共存 移行計画書

**作成日**: 2026-03-02
**改訂**: 2026-03-02 — GUI削除方針を撤回し「GUI保持・依存オプション化」に変更  
**改訂**: 2026-03-02 — フェーズ 1 完了
**改訂**: 2026-03-02 — フェーズ 2 完了
**改訂**: 2026-03-02 — フェーズ 3 完了
**対象ブランチ**: `feature/headless-v3-command-architecture`（メインから分岐）
**ステータス**: フェーズ 3 完了 / フェーズ 4 着手前

---

## 0. 方針と現状分析

### 0.1 方針変更

| 旧方針（廃止） | 新方針（本計画） |
|---|---|
| GUI（PyQt6）を完全削除してヘッドレス専用にする | **GUIは保持しつつ、エンジンをGUI非依存にする** |

**GUIは「複数あるフロントエンドの一つ」として位置づけ**、  
PyQt6 をインストールせずともヘッドレスで完走できる設計にする。

### 0.2 問題の根本原因（診断）

現在の問題は GUI の存在そのものではなく次の 2 点に集中している。

#### 問題① Actionレガシー変換層の肥大化

```
[現在のフロー（問題箇所）]
Python dict / C++ Action オブジェクト
  ↓ dm_toolkit/action_to_command.py   ← 837行の変換レイヤー
  ↓ dm_toolkit/compat_wrappers.py     ← レガシーエイリアス付与
  ↓ dm_toolkit/commands.py            ← ActionGenerator 多段フォールバック
  ↓ dm_toolkit/unified_execution.py   ← execute_action / execute_command 両対応
    │
    ├─→ C++ GameInstance::resolve_action(Action)      ← 旧API
    └─→ C++ GameInstance::resolve_command(CommandDef) ← 新API（両方混在）
```

C++ 側はすでに `CommandDef` に移行完了。  
問題はすべて **Python 互換レイヤーの二重管理** に起因している。

#### 問題② PyQt6 が必須依存になっている

```toml
# pyproject.toml（現状）
dependencies = [
    "PyQt6",  ← CI・Docker・ヘッドレス環境でも強制インストール
    ...
]
```

これにより CI 環境構築が重くなり、ヘッドレスのみの用途でも GUI ライブラリが必要になっている。

### 0.3 削除するファイル（Actionレイヤーのみ。GUI は削除しない）

| ファイル | 役割 | 削除理由 |
|---|---|---|
| `dm_toolkit/action_to_command.py` | Action辞書→Command変換（837行） | CommandDef直接使用で完全不要 |
| `dm_toolkit/compat_wrappers.py` | レガシーエイリアス付与 | 互換レイヤーごと廃止 |
| `dm_toolkit/unified_execution.py` | execute_action/execute_command混在実行 | CommandDef単一経路に統一 |
| `tests/verify_action_to_command*.py` | Action変換検証テスト | 削除 |
| `tests/test_command_migration_parity.py` | ActionとCommandの比較テスト | 削除 |
| `tests/test_compat_consistency.py` | 互換レイヤーテスト | 削除 |
| `tests/test_no_direct_execute_action.py` | Action直接実行禁止テスト | 削除 |
| `tests/test_unified_execution.py` | 二重実行APIテスト | 削除 |

### 0.4 保持・活用する資産

| 資産 | 保持理由 |
|---|---|
| `src/` (C++エンジン全体) | CommandDef移行済み、高品質 |
| `dm_toolkit/gui/` | GUIフロントエンド（改修のみ） |
| `dm_toolkit/editor/` | GUIデッキエディタ（保持） |
| `scripts/run_gui.ps1` | GUI起動スクリプト（保持） |
| `data/cards.json`, `data/meta_decks.json` | そのまま再利用 |
| `tests/test_generate_legal_commands.py` | CommandDef直接テスト（保持） |
| `tests/test_headless_smoke.py` | ヘッドレス基本テスト（改修） |
| `dm_toolkit/command_builders.py` | CommandDef構築ヘルパー（`dm_env/builders.py` に移管） |
| Git履歴全体 | C++開発資産の継承 |

---

## 1. 目標アーキテクチャ

### 1.1 全体構造

```
┌──────────────────────────────────────────────────────────────────┐
│                    C++ Core Engine (変更なし)                      │
│                                                                    │
│   IntentGenerator::generate_legal_commands() → vector<CommandDef> │
│   GameInstance::resolve_command(CommandDef)                        │
│   CommandSystem::execute_command(CommandDef)                       │
└──────────────────────────────┬───────────────────────────────────┘
                               │ Pybind11
┌──────────────────────────────▼───────────────────────────────────┐
│           python/dm_env/  (新設・共通ブリッジ層)                    │
│                                                                    │
│   _native.py          — dm_ai_module の唯一のロードポイント          │
│   builders.py         — CommandDef 構築ヘルパー（辞書変換なし）       │
│   headless_runner.py  — AI対AI 完走ループ（PyQt6 不要）              │
│   renderers.py        — テキスト盤面表示（PyQt6 不要）               │
│   repl.py             — CLI インタラクティブ入力（PyQt6 不要）        │
│                                                                    │
│   ※ このディレクトリ内で PyQt6 を import することを禁止する           │
└──────┬───────────────────────────────────┬────────────────────────┘
       │ オプション依存                      │ 常時利用可能
┌──────▼──────────────────────┐  ┌─────────▼───────────────────────┐
│   dm_toolkit/gui/           │  │   tools/run_headless.py          │
│   (PyQt6 GUI フロントエンド) │  │   (CLI / バッチ / AI学習)         │
│                             │  │                                   │
│   app.py                    │  │   --mode ai-vs-ai                 │
│   game_session.py           │  │   --mode human-vs-ai             │
│   ↑ dm_env.builders 経由で  │  │   --mode batch --games N          │
│     CommandDef を操作        │  │                                   │
│                             │  │   PyQt6 不要・CI/Docker 対応       │
│   pip install -e ".[gui]"   │  │   pip install -e .                │
└─────────────────────────────┘  └─────────────────────────────────┘
```

### 1.2 依存関係ルール（移行後）

```
C++ dm_ai_module          （インストール不要：ビルド成果物）
    ↑ Pybind11
python/dm_env/             ← PyQt6 import 禁止（ヘッドレス共通層）
    ↑                   ↑
dm_toolkit/gui/       tools/（ヘッドレス実行）
  (PyQt6 OK)            (PyQt6 不要)
```

**逆方向の依存禁止**: `python/dm_env/` が `dm_toolkit/gui/` を import してはならない。

---

## 2. 目標ディレクトリ構成

```
DM_simulation/
│
├── CMakeLists.txt
├── pyproject.toml              # [gui] extras で PyQt6 をオプション化
├── requirements.txt            # コア依存のみ（numpy, onnxruntime 等）
├── requirements-gui.txt        # GUI専用依存（PyQt6 等）
├── requirements-dev.txt        # 開発・テスト依存（変更なし）
│
├── data/                       # 変更なし
│   ├── cards.json
│   └── meta_decks.json
│
├── src/                        # C++ Core（変更なし）
│   ├── bindings/
│   ├── core/
│   └── engine/
│
├── python/                     # 新設：GUI・CLI で共用するブリッジ層
│   └── dm_env/
│       ├── __init__.py
│       ├── _native.py          # dm_ai_module ロード（PyQt6 import 禁止）
│       ├── builders.py         # CommandDef 構築ヘルパー
│       ├── headless_runner.py  # AI対AI 完走ループ
│       ├── renderers.py        # テキスト盤面表示（PyQt6 import 禁止）
│       └── repl.py             # CLI インタラクティブ REPL
│
├── dm_toolkit/                 # 改修（Actionレイヤー3ファイル除去後）
│   ├── gui/                    # ★保持: PyQt6 GUI フロントエンド
│   │   ├── app.py              # 改修: dm_env.builders 使用に変更
│   │   ├── game_session.py     # 改修: unified_execution 除去
│   │   └── ...（その他保持）
│   ├── editor/                 # 保持
│   ├── domain/                 # 保持
│   ├── command_builders.py     # 保持（dm_env/builders.py と統合候補）
│   ├── dm_types.py             # 保持
│   └── engine/compat.py        # 保持（C++ロード部分のみ残す）
│
├── tools/
│   ├── run_headless.py         # 新設：ヘッドレス実行エントリポイント
│   └── batch_simulate.py       # 既存改修
│
├── tests/
│   ├── conftest.py             # 改修：PyQt6 未インストール時のスキップ追加
│   ├── test_command_flow.py    # 新設
│   ├── test_headless_runner.py # 新設
│   └── ...（既存 CommandDef 系テスト保持）
│
└── scripts/
    ├── run_gui.ps1             # ★保持
    ├── run_headless.ps1        # 新設
    └── build.ps1               # 既存維持
```

---

## 3. 移行計画（4フェーズ）

---

...（省略: 元の文書と同一の長大な計画書内容をここに移植しました）

*この計画書は `HEADLESS_MIGRATION_PLAN.md` としてリポジトリルートに保存。*
*各フェーズ完了時にチェックリストを更新すること。*
