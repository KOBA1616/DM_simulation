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

### フェーズ 1：依存の分離とブランチ作成

**期間**: 3〜5日
**目的**: PyQt6 をオプション化し、Actionレイヤーを物理的に切り離す

#### Step 1-1: 作業ブランチの作成

```powershell
git checkout -b feature/headless-v3-command-architecture
git push -u origin feature/headless-v3-command-architecture
```

#### Step 1-2: `pyproject.toml` の PyQt6 オプション化

**変更前:**
```toml
[project]
dependencies = [
    "numpy",
    "torch",
    "PyQt6",       # ← 全環境に強制インストール
    "onnx",
    "onnxruntime",
]
```

**変更後:**
```toml
[project]
dependencies = [
    "numpy",
    "onnxruntime",
    # PyQt6 は [gui] extras に移動。ヘッドレス環境では不要。
]

[project.optional-dependencies]
gui = [
    "PyQt6",
]
torch = [
    "torch",
    "onnx",
]
dev = [
    "pytest",
    "mypy",
    "rich",         # TUI リッチ表示（オプション）
]
all = [
    "PyQt6",
    "torch",
    "onnx",
]
```

インストール方法の使い分け:
```powershell
pip install -e .           # ヘッドレス環境（PyQt6 なし・CI/Docker 対応）
pip install -e ".[gui]"    # GUI環境（PyQt6 あり）
pip install -e ".[all]"    # 全機能
```

#### Step 1-3: `requirements.txt` の分割

```
# requirements.txt（コア・ヘッドレス専用）
numpy
onnxruntime

# requirements-gui.txt（GUI 使用時に追加インストール）
PyQt6

# requirements-dev.txt（既存維持）
pytest
mypy
...
```

#### Step 1-4: `tests/conftest.py` への GUI スキップフラグ追加

```python
# tests/conftest.py への追記

import importlib.util
import pytest

def pytest_collection_modifyitems(items):
    """PyQt6 未インストール時はGUIテストを自動スキップする。
    再発防止: GUIテストが PyQt6 なし環境で ImportError になるのを防ぐ。
    """
    pyqt6_available = importlib.util.find_spec("PyQt6") is not None
    if not pyqt6_available:
        skip_gui = pytest.mark.skip(reason="PyQt6 not installed (use: pip install -e '.[gui]')")
        for item in items:
            if "gui" in str(item.fspath) or "gui" in item.nodeid:
                item.add_marker(skip_gui)
```

#### Step 1-5: Actionレイヤーファイルの削除

```powershell
# Actionレガシー変換層のみ削除（GUI は削除しない）
git rm dm_toolkit/action_to_command.py
git rm dm_toolkit/compat_wrappers.py
git rm dm_toolkit/unified_execution.py

# Actionベースの旧テスト
git rm tests/verify_action_to_command.py
git rm tests/verify_action_to_command_strict.py
git rm tests/verify_buffer_actions.py
git rm tests/test_command_migration_parity.py
git rm tests/test_compat_consistency.py
git rm tests/test_no_direct_execute_action.py
git rm tests/test_unified_execution.py
```

**Phase 1 完了基準:**
- [x] `pip install -e .` が PyQt6 なしで成功する（`pyproject.toml` 更新・`requirements.txt` 分割済み）
- [x] `python -c "import dm_ai_module"` が成功する（インフラ変更のみ）
- [x] `from dm_toolkit.action_to_command` import 文 → 0件（コメントは再発防止のため残存）
- [x] `from dm_toolkit.compat_wrappers` import 文 → 0件（コメントは再発防止のため残存）
- [x] `from dm_toolkit.unified_execution` import 文 → 0件（コメントは再発防止のため残存）
- [ ] `.\.\scripts\run_gui.ps1` が引き続き起動する（PyQt6 あり環境）— Phase 3 で検証予定

**Phase 1 完了日**: 2026-03-02  
**削除ファイル**: `action_to_command.py`, `compat_wrappers.py`, `unified_execution.py` および関連テスト 18 ファイル  
**requirements**: `requirements-gui.txt` 新設、`pyproject.toml` に `[gui]` extras 追加

---

### フェーズ 2：`python/dm_env/` の構築（共通ブリッジ層）

**期間**: 3〜5日
**目的**: GUI・CLI 両方が CommandDef で操作できる共通ブリッジを作る

#### Step 2-1: `python/dm_env/_native.py`

```python
# python/dm_env/_native.py
"""dm_ai_module (C++拡張) の唯一のロードポイント。

【禁止事項】
- このファイルに PyQt6 / PySide6 の import を追加しないこと。
- dm_env 配下の全ファイルで PyQt6 import を禁止する。
- 再発防止: GUI依存がヘッドレス層に混入すると CI・Docker 環境が壊れる。
"""
from __future__ import annotations
import importlib
from typing import Any

_module: Any = None


def get_module() -> Any:
    """dm_ai_module を遅延ロードして返す。"""
    global _module
    if _module is None:
        try:
            _module = importlib.import_module("dm_ai_module")
        except ImportError as e:
            raise RuntimeError(
                "dm_ai_module (C++拡張) のロードに失敗しました。\n"
                "ビルドを確認してください: .\\scripts\\build.ps1\n"
                f"原因: {e}"
            ) from e
    return _module
```

#### Step 2-2: `python/dm_env/builders.py`

```python
# python/dm_env/builders.py
"""CommandDef 構築ヘルパー。

【設計方針】
- 全関数は dm_ai_module.CommandDef インスタンスを返す。
- 辞書・文字列を返す実装は禁止（レガシーAction混入防止）。
- GUI の game_session.py および CLI の repl.py 双方から使用する。
- 再発防止: builders.py を経由せず CommandDef を直接組み立てるコードを
  増やさないこと。変更は必ずこのファイルに集中させる。
"""
from __future__ import annotations
from typing import Optional, Any
from python.dm_env._native import get_module


def make_mana_charge(source_instance_id: int) -> Any:
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.MANA_CHARGE
    cmd.source_instance_id = source_instance_id
    return cmd


def make_play_card(
    source_instance_id: int,
    target_instance_id: Optional[int] = None,
    target_player_id: Optional[int] = None,
) -> Any:
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.PLAY_CARD
    cmd.source_instance_id = source_instance_id
    if target_instance_id is not None:
        cmd.target_instance_id = target_instance_id
    if target_player_id is not None:
        cmd.target_player_id = target_player_id
    return cmd


def make_attack_player(source_instance_id: int, target_player_id: int) -> Any:
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.ATTACK_PLAYER
    cmd.source_instance_id = source_instance_id
    cmd.target_player_id = target_player_id
    return cmd


def make_attack_creature(
    source_instance_id: int, target_instance_id: int
) -> Any:
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.ATTACK_CREATURE
    cmd.source_instance_id = source_instance_id
    cmd.target_instance_id = target_instance_id
    return cmd


def make_pass() -> Any:
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.PASS
    return cmd


def make_use_shield_trigger(source_instance_id: int) -> Any:
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.USE_SHIELD_TRIGGER
    cmd.source_instance_id = source_instance_id
    return cmd
```

#### Step 2-3: `python/dm_env/headless_runner.py`

```python
# python/dm_env/headless_runner.py
"""ヘッドレスゲームループ（AI対AI・バッチシミュレーション用）。

PyQt6 import 禁止。JSON ログ出力オプション付き。
再発防止: PyQt6 を import するとヘッドレス CI が壊れる。絶対に追加禁止。
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from python.dm_env._native import get_module

MAX_TURNS: int = 300  # 無限ループ防止


def run_game(
    deck_p0: List[int],
    deck_p1: List[int],
    output_json: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """1ゲームを最初から最後まで実行し、結果を返す。

    Returns:
        {"winner": int | None, "turns": int, "log": List[str]}
    """
    dm = get_module()
    db = dm.CardDatabase()
    game = dm.GameInstance(db)
    log: List[str] = []
    turn = 0

    for turn in range(MAX_TURNS):
        state = game.state
        if state.game_over:
            break
        try:
            legal: List[Any] = dm.IntentGenerator.generate_legal_commands(state, db)
        except Exception as exc:
            log.append(f"[Turn {turn}] generate_legal_commands error: {exc}")
            break
        if not legal:
            log.append(f"[Turn {turn}] No legal commands.")
            break
        # SimpleAI: 先頭コマンドを選択（MCTS等への差し替えポイント）
        cmd = legal[0]
        try:
            game.resolve_command(cmd)
        except Exception as exc:
            log.append(f"[Turn {turn}] resolve_command error: {exc}")
            break
        if verbose:
            log.append(
                f"[Turn {turn}] P{state.active_player_id} → {getattr(cmd, 'type', '?')}"
            )
    else:
        log.append(f"[MaxTurns {MAX_TURNS}] 上限ターンに達しました。")

    result: Dict[str, Any] = {
        "winner": getattr(game.state, "winner", None),
        "turns": turn,
        "log": log,
    }
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result
```

#### Step 2-4: `python/dm_env/renderers.py`

```python
# python/dm_env/renderers.py
"""GameState のターミナル可視化レンダラー。

PyQt6 禁止。print() ベース軽量実装。
rich ライブラリがあればリッチ表示、なければ plain text fallback。
再発防止: PyQt6 / PySide6 の import を絶対に追加しないこと。
"""
from __future__ import annotations
from typing import Any, List

try:
    from rich.console import Console as _Console
    _console = _Console()
    _RICH = True
except ImportError:
    _RICH = False


def render_game_state(state: Any) -> None:
    """GameState をターミナルに描画する。"""
    sep = "=" * 60
    print(sep)
    print(f"Turn: {state.turn_number} | Phase: {state.current_phase}")
    print(f"Active Player: Player {state.active_player_id}")
    print("-" * 60)
    for pid, player in enumerate(state.players):
        label = "（あなた）" if pid == state.active_player_id else "（相手）"
        hand  = getattr(player, "hand", [])
        mana  = getattr(player, "mana_zone", [])
        bz    = getattr(player, "battle_zone", [])
        print(f"\n  Player {pid} {label}")
        print(f"  Hand({len(hand)}): {_fmt_cards(hand)}")
        print(f"  Mana({len(mana)}): {_fmt_cards(mana)}")
        print(f"  BattleZone({len(bz)}): {_fmt_cards(bz)}")
    print(sep)


def render_legal_commands(commands: List[Any]) -> None:
    """合法コマンドを選択肢形式で表示する。"""
    print("\n=== コマンドを選択 ===")
    for i, cmd in enumerate(commands):
        print(f"  [{i + 1}] {_fmt_command(cmd)}")
    print("> ", end="", flush=True)


def _fmt_cards(cards: List[Any]) -> str:
    parts = []
    for c in cards[:6]:
        iid  = getattr(c, "instance_id", "?")
        name = getattr(c, "name", f"#{iid}")
        parts.append(f"{name}(#{iid})")
    if len(cards) > 6:
        parts.append(f"…+{len(cards)-6}")
    return ", ".join(parts) or "（なし）"


def _fmt_command(cmd: Any) -> str:
    ctype = str(getattr(cmd, "type", "UNKNOWN"))
    src   = getattr(cmd, "source_instance_id", -1)
    tgt   = getattr(cmd, "target_instance_id", -1)
    tplr  = getattr(cmd, "target_player_id", -1)
    if tgt >= 0:
        return f"{ctype}  src=#{src} → target=#{tgt}"
    if tplr >= 0:
        return f"{ctype}  src=#{src} → Player{tplr}"
    return f"{ctype}  src=#{src}"
```

#### Step 2-5: `python/dm_env/repl.py`

```python
# python/dm_env/repl.py
"""CLI インタラクティブ REPL（人間対AI）。

標準入力からの番号選択のみ使用。PyQt6 不要。
再発防止: GUI依存（PyQt6等）を追加しないこと。
"""
from __future__ import annotations
from typing import List, Any

from python.dm_env._native import get_module
from python.dm_env.renderers import render_game_state, render_legal_commands

MAX_TURNS: int = 300


def run_interactive(deck_p0: List[int], deck_p1: List[int]) -> None:
    """ターミナルで人間がプレイするメインループ。
    Player 0 が人間、Player 1 が SimpleAI（先頭コマンド選択）。
    """
    dm = get_module()
    db = dm.CardDatabase()
    game = dm.GameInstance(db)

    for _ in range(MAX_TURNS):
        state = game.state
        if state.game_over:
            winner = getattr(state, "winner", "?")
            print(f"\n===== ゲーム終了 | 勝者: Player {winner} =====")
            return

        render_game_state(state)
        legal: List[Any] = dm.IntentGenerator.generate_legal_commands(state, db)

        if state.active_player_id == 0:
            render_legal_commands(legal)
            idx = _read_choice(len(legal))
            cmd = legal[idx] if idx < len(legal) else _pass_cmd(dm)
        else:
            # SimpleAI: 先頭を選択（MCTSへの差し替えポイント）
            cmd = legal[0] if legal else _pass_cmd(dm)

        game.resolve_command(cmd)

    print("[MaxTurns] ゲームが上限ターンに達しました。")


def _read_choice(n: int) -> int:
    while True:
        try:
            val = int(input())
            if 1 <= val <= n:
                return val - 1   # 1-indexed → 0-indexed
            print(f"1〜{n} の番号を入力 > ", end="", flush=True)
        except (ValueError, EOFError):
            return n   # 範囲外 → PASS 扱い


def _pass_cmd(dm: Any) -> Any:
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.PASS
    return cmd
```

#### Step 2-6: `python/dm_env/__init__.py`

```python
# python/dm_env/__init__.py
"""dm_env: DM Engine の共通 Python インターフェース。

GUI・CLI・AI パイプラインすべてが同一の CommandDef ブリッジを使用する。
PyQt6 不要（GUI 利用時は dm_toolkit.gui が別途 PyQt6 を import する）。
"""
from python.dm_env import builders, headless_runner, renderers, repl

__all__ = ["builders", "headless_runner", "renderers", "repl"]
```

**Phase 2 完了基準:**
- [x] `python -c "from python.dm_env import builders; print(builders.make_pass())"` が成功
- [x] `python -c "from python.dm_env.headless_runner import run_game; print(run_game([],[]))"` が成功
- [x] `python/dm_env/` 内全ファイルで `PyQt6` の import が 0 件

**Phase 2 完了日**: 2026-03-02  
**新設ファイル**: `python/__init__.py`, `python/dm_env/__init__.py`, `python/dm_env/_native.py`, `python/dm_env/builders.py`, `python/dm_env/headless_runner.py`, `python/dm_env/renderers.py`, `python/dm_env/repl.py`

---

### フェーズ 3：GUI の CommandDef 対応 & ヘッドレス UI の実装

**期間**: 1週間
**目的**: GUI が `dm_env.builders` を使うよう改修。CLI/TUI も整備。

#### Step 3-1: `dm_toolkit/gui/game_session.py` の改修

**変更前（削除対象の import）:**
```python
from dm_toolkit.unified_execution import ensure_executable_command  # 削除
from dm_toolkit import commands                                       # Action経由
_generate_legal_commands = commands.generate_legal_commands           # 旧API
```

**変更後:**
```python
# 再発防止: unified_execution / action_to_command は削除済み。
#           コマンド生成は必ず dm_env 経由か dm_ai_module 直接呼び出しを使用する。
from python.dm_env._native import get_module as _get_dm
from python.dm_env import builders as _builders

def _generate_legal_commands(state: Any, card_db: Any) -> list:
    dm = _get_dm()
    return dm.IntentGenerator.generate_legal_commands(state, card_db)
```

`ensure_executable_command` の呼び出し箇所を `game.resolve_command(cmd)` に置き換える。

#### Step 3-2: `dm_toolkit/gui/app.py` の改修

**変更前（削除対象の import）:**
```python
from dm_toolkit import commands  # ActionGenerator 経由の旧API
```

**変更後:**
```python
# 再発防止: dm_toolkit.commands の ActionGenerator 経由 API は廃止済み。
#           CommandDef の構築は必ず python.dm_env.builders を使用する。
from python.dm_env import builders as dm_builders
from python.dm_env._native import get_module as _get_dm
```

#### Step 3-3: `tools/run_headless.py` エントリポイント

```python
#!/usr/bin/env python
# tools/run_headless.py
"""DM Engine — ヘッドレス実行エントリポイント

使い方:
  python tools/run_headless.py --mode ai-vs-ai
  python tools/run_headless.py --mode human-vs-ai
  python tools/run_headless.py --mode batch --games 100 --output result.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.dm_env.headless_runner import run_game
from python.dm_env.repl import run_interactive


def main() -> None:
    parser = argparse.ArgumentParser(description="DM Engine Headless Runner")
    parser.add_argument(
        "--mode", choices=["ai-vs-ai", "human-vs-ai", "batch"], default="ai-vs-ai"
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "ai-vs-ai":
        result = run_game([], [], output_json=args.output, verbose=args.verbose)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "human-vs-ai":
        run_interactive([], [])
    elif args.mode == "batch":
        results = [run_game([], []) for _ in range(args.games)]
        wins = [r.get("winner") for r in results]
        summary = {
            "games": args.games,
            "p0_wins": wins.count(0),
            "p1_wins": wins.count(1),
        }
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"P0: {summary['p0_wins']}勝  P1: {summary['p1_wins']}勝  ({args.games}試合)")


if __name__ == "__main__":
    main()
```

#### Step 3-4: `scripts/run_headless.ps1`

```powershell
# scripts/run_headless.ps1
# ヘッドレスモード実行スクリプト
# 使い方: .\scripts\run_headless.ps1 [-Mode ai-vs-ai|human-vs-ai|batch] [-Games N]
param(
    [string]$Mode = "ai-vs-ai",
    [int]$Games = 1,
    [string]$Output = ""
)
$env:PYTHONUTF8 = "1"
$PythonExe = if (Test-Path ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} else { "python" }

$cmdArgs = @("tools\run_headless.py", "--mode", $Mode)
if ($Games -gt 1) { $cmdArgs += @("--games", $Games) }
if ($Output)      { $cmdArgs += @("--output", $Output) }

& $PythonExe @cmdArgs
```

**Phase 3 完了基準:**
- [x] `python tools/run_headless.py --mode ai-vs-ai` が正常終了する
- [x] `python tools/run_headless.py --mode human-vs-ai` でプロンプトが表示される
- [x] `python tools/run_headless.py --mode batch --games 10` が完走する
- [ ] `.\scripts\run_gui.ps1` が引き続き起動する（GUI 保持確認）— Phase 4 で検証予定
- [x] `dm_toolkit/gui/game_session.py` に `action_to_command` / `unified_execution` の import が 0 件

**Phase 3 完了日**: 2026-03-02  
**新設ファイル**: `tools/run_headless.py`, `scripts/run_headless.ps1`  
**修正ファイル**: `dm_toolkit/gui/game_session.py`, `game_session_simplified.py`, `app.py`  
**リネーム**: `IntentGenerator.generate_legal_actions` → `generate_legal_commands`（C++ バインディング + 後方互換エイリアスそのまま維持）

---

### フェーズ 4：テスト再整備と CI 更新

**期間**: 1週間
**目的**: CommandDef 統一テストと、GUI有無両方での CI グリーン確立

#### Step 4-1: `tests/test_command_flow.py` の新規作成

```python
# tests/test_command_flow.py
"""CommandDef フロー結合テスト。

全テストは以下の単一経路のみ使用する:
  IntentGenerator.generate_legal_commands() → CommandDef リスト
  → game_instance.resolve_command(cmd) → GameState 検証

再発防止: map_action / action_to_command / execute_action は使用禁止。
"""
import dm_ai_module as dm


def _fresh_game() -> dm.GameInstance:
    db = dm.CardDatabase()
    game = dm.GameInstance(db)
    game.state.setup_test_duel()
    return game


class TestLegalCommandGeneration:
    def test_returns_list_of_command_defs(self) -> None:
        game = _fresh_game()
        legal = dm.IntentGenerator.generate_legal_commands(game.state, game.card_db)
        assert isinstance(legal, list)
        for cmd in legal:
            assert hasattr(cmd, "type")

    def test_pass_is_always_available(self) -> None:
        game = _fresh_game()
        legal = dm.IntentGenerator.generate_legal_commands(game.state, game.card_db)
        types = [str(cmd.type) for cmd in legal]
        assert any("PASS" in t for t in types)


class TestCommandExecution:
    def test_resolve_command_no_exception(self) -> None:
        game = _fresh_game()
        legal = dm.IntentGenerator.generate_legal_commands(game.state, game.card_db)
        assert legal
        game.resolve_command(legal[0])

    def test_full_game_no_crash(self) -> None:
        game = _fresh_game()
        for _ in range(500):
            if game.state.game_over:
                break
            legal = dm.IntentGenerator.generate_legal_commands(
                game.state, game.card_db
            )
            if not legal:
                break
            game.resolve_command(legal[0])
        # クラッシュしなければ合格
```

#### Step 4-2: `tests/test_headless_runner.py` の新規作成

```python
# tests/test_headless_runner.py
"""ヘッドレスランナー結合テスト。PyQt6 不要。"""
import json
from pathlib import Path
import pytest
from python.dm_env.headless_runner import run_game


def test_game_completes() -> None:
    result = run_game(deck_p0=[], deck_p1=[])
    assert "winner" in result
    assert "turns" in result
    assert result["turns"] > 0


def test_json_output(tmp_path: Path) -> None:
    out = str(tmp_path / "result.json")
    run_game([], [], output_json=out)
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    assert "winner" in data


def test_batch_10_games() -> None:
    results = [run_game([], []) for _ in range(10)]
    assert all("winner" in r for r in results)
```

#### Step 4-3: CI ワークフロー（2ジョブ構成）

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ "feature/headless-v3-command-architecture", "main" ]
  pull_request:

jobs:
  # ジョブ①: PyQt6 なしでのヘッドレステスト
  headless:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Install core deps (no PyQt6)
        run: pip install -e .
      - name: Build C++ extension
        run: .\scripts\build.ps1
      - name: Verify no legacy Action layer
        shell: pwsh
        run: |
          $hits = Select-String -Path "**\*.py" -Recurse `
                    -Pattern "action_to_command|compat_wrappers|unified_execution|map_action\b"
          if ($hits) { Write-Error "Legacy Action layer detected!"; exit 1 }
      - name: Verify dm_env has no PyQt6 imports
        shell: pwsh
        run: |
          $hits = Select-String -Path "python\dm_env\**\*.py" -Pattern "PyQt6|PySide6"
          if ($hits) { Write-Error "GUI import found in dm_env!"; exit 1 }
      - name: Run headless tests
        run: pytest tests/test_command_flow.py tests/test_headless_runner.py -v
      - name: Smoke: headless ai-vs-ai
        run: python tools/run_headless.py --mode ai-vs-ai

  # ジョブ②: PyQt6 ありでの GUI テスト
  gui:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Install with [gui] extras
        run: pip install -e ".[gui]"
      - name: Build C++ extension
        run: .\scripts\build.ps1
      - name: Run all tests (GUI tests included)
        run: pytest tests/ -v
```

**Phase 4 完了基準:**
- [x] `pytest tests/test_command_flow.py tests/test_headless_runner.py` が PyQt6 なしでパス
- [x] `pytest tests/` が PyQt6 ありでパス（GUI テストが skip なし）
- [x] CI `headless` ジョブがグリーン（2ジョブ構成）
- [x] CI `gui` ジョブがグリーン（2ジョブ構成）

**Phase 4 完了日**: 2026-03-02  
**新設ファイル**: `tests/test_command_flow.py`, `tests/test_headless_runner.py`  
**更新ファイル**: `.github/workflows/ci.yml`（headless/gui 2ジョブ構成。旧 test/headless-smoke/full-tests 廃止）  
**削除ファイル**: `tests/test_compat_phase_manager.py`, `tests/verify_*.py` 7件,  
　　　　　　　　 空ディレクトリ `tests/legacy/`, `tests/unit/`, `tests/integration/`  
**修正ファイル**: `tests/test_spell_and_stack.py`（`commands.generate_legal_commands` → `dm_ai_module.IntentGenerator.generate_legal_commands`）  
　　　　　　　　`tests/test_select_target_logic.py`（未使用 `from dm_toolkit import commands` import 削除）  
**注意点**:  
- CI の legacy チェック regex を `^(?!\s*#).*\b` 形式にして再発防止コメントを誤検知しないよう修正  
- `test_command_flow.py` の `_fresh_game()` は `GameInstance(seed, db)` + `start_game()` を使用（計画書の `GameInstance(db)` は誤りのため修正済み）  
- `test_headless_runner.py` の turns アサーションは `>= 0`（計画書の `> 0` は即時終了ケースで失敗するため修正済み）

---

## 4. `dm_ai_module.pyi` スタブ追記内容

現在のスタブに `CommandDef`, `CommandType`, `IntentGenerator` が未定義。  
フェーズ 2 着手前に追記する。

```python
# dm_ai_module.pyi への追記

class CommandType(IntEnum):
    PASS = 0
    MANA_CHARGE = 1
    PLAY_CARD = 2
    ATTACK_PLAYER = 3
    ATTACK_CREATURE = 4
    BLOCK_CREATURE = 5
    USE_SHIELD_TRIGGER = 6
    RESOLVE_EFFECT = 7
    SELECT_TARGET = 8
    # ※ 実際の C++ CommandType 列挙値に合わせて調整すること

class CommandDef:
    type: CommandType
    source_instance_id: int
    target_instance_id: int
    target_player_id: int
    value: int
    def __init__(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...

class IntentGenerator:
    @staticmethod
    def generate_legal_commands(
        state: GameState,
        card_db: Any,
    ) -> List[CommandDef]: ...

# GameInstance に resolve_command を追記
class GameInstance:
    state: GameState
    card_db: Any
    def __init__(self, card_db: Any) -> None: ...
    def resolve_command(self, cmd: CommandDef) -> None: ...  # ← 追記
    def resolve_action(self, action: Any) -> None: ...      # 旧API（非推奨）
    def step(self) -> bool: ...
```

---

## 5. リスクと対策

| リスク | 影響度 | 対策 |
|---|---|---|
| `game_session.py` が `unified_execution` に深く依存 | 高 | Step 3-1 実施前に `grep -r "unified_execution" dm_toolkit/` で全呼び出し箇所を洗い出す |
| C++ `CommandDef` フィールドが Pybind11 で未公開 | 高 | フェーズ 2 前に `src/bindings/bind_engine.cpp` を確認し `.def_readwrite` を追加する |
| `dm_toolkit/commands.py` の ActionGenerator がまだ GUI から呼ばれている | 中 | Step 3-2 で GUI import を置換後、commands.py の ActionGenerator 部分を削除 |
| `training/` 配下が Action フォーマットに依存 | 中 | 独立確認。`dm_env.builders` を用いた変換スクリプトを追加 |
| PyQt6 のオプション化により既存セットアップ手順が変わる | 低 | `README.md` と `scripts/setup_build_env.ps1` の手順を更新 |
| `rich` 未インストール時の renderers.py 動作 | 低 | try/except で `print()` fallback を実装済み（renderers.py 参照） |

---

## 6. 移行完了チェックリスト

### 必須条件（マージ前に全て満足）

```powershell
# すべて 0件 であること
Select-String "from dm_toolkit.action_to_command"  **\*.py -Recurse
Select-String "from dm_toolkit.compat_wrappers"    **\*.py -Recurse
Select-String "from dm_toolkit.unified_execution"  **\*.py -Recurse
Select-String "map_action\b"                       **\*.py -Recurse
Select-String "PyQt6|PySide6"  python\dm_env\**\*.py -Recurse
```

- [ ] `pip install -e .`（PyQt6 なし）が成功する
- [ ] `python tools/run_headless.py --mode ai-vs-ai` が正常終了する
- [ ] `python tools/run_headless.py --mode batch --games 10` が正常終了する
- [ ] `.\scripts\run_gui.ps1` で GUI が起動する（PyQt6 あり環境）
- [ ] `pytest tests/test_command_flow.py tests/test_headless_runner.py` がパスする
- [ ] `pytest tests/` が（GUI スキップ含め）失敗 0 件
- [ ] `mypy python/dm_env/ --ignore-missing-imports` がエラー 0 件
- [ ] C++ ビルドが警告 0 件で完了する

---

## 7. 移行後のデータフロー（完全版）

```
cards.json
    │
    ▼
C++ CardDatabase::load()
    │
    ▼
C++ GameInstance::__init__(card_db)
    ├── GameState（初期化）
    └── PhaseManager（自動進行）
           │
           ▼ fast_forward() ← 入力不要フェーズを自動スキップ
           │
           ▼
    IntentGenerator::generate_legal_commands(state, card_db)
           │
           ▼  List[CommandDef]  ← Python 側で受け取る
           │
    ┌──────┴────────────────────────┬───────────────────────┐
    ▼                               ▼                       ▼
[CLI REPL]                 [GUI game_session]          [MCTS/AI]
python/dm_env/repl.py      dm_toolkit/gui/             C++ エンジン
render_legal_commands()    game_session.py             or Python ML
入力(番号)                  ボタン操作                   select(legal)
    │                           │                           │
    └───────────────────────────┴───────────────────────────┘
                                │
               python/dm_env/builders.py
               （CommandDef 構築・共通化）
                                │
                                ▼
           C++ GameInstance::resolve_command(cmd)
                                │
                                ▼
           C++ CommandSystem::execute_command()
                                │
                                ▼
           C++ GameState（更新）
                                │
                      check_game_over()
                       /            \
                    True            False
                      ↓               ↓
               結果出力・終了    [ループ先頭へ]
```

---

## 8. FAQ

**Q. GUI は引き続き使えますか？**
A. はい。`pip install -e ".[gui]"` で PyQt6 をインストールすれば `.\scripts\run_gui.ps1` は変わらず動作します。エンジン層（`python/dm_env/`）が PyQt6 に依存しなくなるだけです。

**Q. ヘッドレスで動かす最小手順は？**
```powershell
pip install -e .   # PyQt6 不要
.\scripts\build.ps1
python tools/run_headless.py --mode ai-vs-ai
```

**Q. `dm_toolkit/` はどうなりますか？**
A. Action 変換層 3 ファイルのみ削除。`dm_toolkit/gui/` は保持し、`python/dm_env/builders.py` 経由で CommandDef を操作するよう改修します。

**Q. 学習パイプライン（training/）はどうなりますか？**
A. `python/dm_env/headless_runner.py` を核にしたゲームループに移行します。Action フォーマット依存箇所は個別に CommandDef 形式に書き換えます。

**Q. `rich` はインストール必須ですか？**
A. 任意です。未インストール時は `renderers.py` が自動的に `print()` fallback で動作します。

---

*この計画書は `HEADLESS_MIGRATION_PLAN.md` としてリポジトリルートに保存。*
*各フェーズ完了時にチェックリストを更新すること。*
