````markdown
# GUI ヘッドレステスト環境構築ガイド

GUI表示環境がない場合（CI環境やヘッドレスサーバー）でもGUI関連のコードを網羅的にテストできる環境の構築方法を説明します。

## 概要

このプロジェクトでは、PyQt6/PySide6のGUIコンポーネントをスタブ（ダミー実装）に置き換えることで、実際のGUI環境がなくてもテストを実行できます。

### アーキテクチャ

```
┌─────────────────────────────────────────────┐
│  run_pytest_with_pyqt_stub.py              │
│  (メインエントリーポイント)                   │
│  - GUI スタブのセットアップ                   │
│  - pytest の起動                            │
└─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────┐
│  python/tests/conftest.py                   │
│  - pytest がテスト収集時に自動的に実行          │
│  - _setup_minimal_gui_stubs() による          │
│    最小限のスタブセットアップ                   │
└─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────┐
│  StubFinder + StubLoader (MetaPathFinder)  │
│  - import フックメカニズム                    │
│  - PyQt6/PySide6 モジュールの動的注入          │
└─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────┐
│  Dummy Classes (スタブ実装)                  │
│  - DummyQWidget                            │
│  - DummyQMainWindow                        │
│  - DummyQApplication                       │
│  - その他の Qt クラス                        │
└─────────────────────────────────────────────┘
```

## 環境構築手順

### 1. 前提条件の確認

```bash
# Python 環境の確認
python --version  # Python 3.8+ が必要

# 必要なパッケージのインストール
pip install -r requirements-dev.txt
```

### 2. スタビングインフラの検証

最初にスタビング機能が正しく動作することを確認します：

```bash
# Windows (PowerShell)
python run_pytest_with_pyqt_stub.py python/tests/gui/test_gui_stubbing.py -v

# Linux/Mac
python3 run_pytest_with_pyqt_stub.py python/tests/gui/test_gui_stubbing.py -v
```

**期待される出力:**
```
[STUB] GUI libraries mocked for headless execution...
[RUN] Starting pytest with args: ['python/tests/gui/test_gui_stubbing.py', '-v']
====== test session starts ======
python/tests/gui/test_gui_stubbing.py::test_gui_libraries_are_stubbed PASSED

[OK] PyQt6 stubbing verified successfully.
====== 1 passed in 0.XX s ======
```

### 3. ヘッドレス環境でのテスト実行

#### 全テストの実行

```bash
python run_pytest_with_pyqt_stub.py
```

#### 特定のテストファイル/ディレクトリの実行

```bash
# GUI関連のテストのみ
python run_pytest_with_pyqt_stub.py python/tests/gui/ -v

# 特定のテストファイル
python run_pytest_with_pyqt_stub.py python/tests/test_your_module.py

# 特定のテスト関数
python run_pytest_with_pyqt_stub.py python/tests/test_your_module.py::test_function_name
```

#### pytest オプションの使用

```bash
# 詳細出力
python run_pytest_with_pyqt_stub.py -v

# 失敗したテストのみ再実行
python run_pytest_with_pyqt_stub.py --lf

# カバレッジ測定
python run_pytest_with_pyqt_stub.py --cov=dm_toolkit --cov-report=html

# 並列実行 (pytest-xdist が必要)
python run_pytest_with_pyqt_stub.py -n auto
```

## スタビングのメカニズム

### 1. import フックによる動的置き換え

`StubFinder` (MetaPathFinder) を使って、`PyQt6` や `PySide6` のインポート時にスタブモジュールを返します：

```python
# sys.meta_path に挿入されたカスタムファインダー
class StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in self.mocks:
            return ModuleSpec(fullname, StubLoader(self.mocks[fullname]))
        return None
```

### 2. ダミークラスと機能的なシグナル実装

実際のGUIクラスを模倣したダミークラスと、**機能的なシグナル/スロット機構**を提供：

```python
class MockSignal:
    """実際に動作するシグナル/スロット実装"""
    def __init__(self):
        self._slots = []
    
    def connect(self, slot):
        self._slots.append(slot)
        return None
    
    def disconnect(self, slot=None):
        if slot is None:
            self._slots.clear()
        elif slot in self._slots:
            self._slots.remove(slot)
        return None
    
    def emit(self, *args, **kwargs):
        # 接続された全てのスロットを実行
        for slot in self._slots:
            slot(*args, **kwargs)
        return None

class EnhancedButton(DummyQWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clicked = MockSignal()  # 機能的なシグナル
```

### 3. サポートされているインタラクション

#### ✅ 完全にサポート

- **ボタンクリック**: `QPushButton.clicked` シグナル
- **コンボボックス選択**: `QComboBox.currentIndexChanged` シグナル
- **テキスト変更**: `QLineEdit.textChanged` シグナル
- **チェックボックス状態**: `QCheckBox.stateChanged` シグナル
- **シグナル/スロット接続**: `connect()`, `disconnect()`, `emit()`
- **イベントハンドラーのエラー検出**: try/except でキャッチ可能
- **複数ウィジェット間の相互作用**: 状態管理とイベントチェーン

#### ⚠️ 部分的にサポート

- **レイアウト**: ウィジェットの追加は可能だが、実際の配置計算は行われない
- **ウィジェット階層**: 親子関係の設定は可能だが、イベント伝播はシミュレート必要

#### ❌ サポート外（実際のGUI環境が必要）

- **視覚的レンダリング**: ピクセルレベルの描画、スクリーンショット
- **実際のユーザー入力**: マウスクリック、キーボード入力の物理的シミュレーション
- **レイアウト計算**: `geometry()`, `size()` の実際の値
- **イベントループ**: `QApplication.exec()` の実行

## 新しいGUIテストの作成

### テストファイルの配置

```
python/tests/
├── gui/                          # GUI関連のテスト
│   ├── test_gui_stubbing.py       # スタビング検証テスト
│   ├── test_gui_interactions.py   # インタラクションテスト例
│   ├── test_your_gui.py           # 新しいGUIテスト
│   └── __pycache__/
└── conftest.py                   # 共通設定とスタブセットアップ
```

### テストの書き方

#### 基本的なウィジェット作成テスト

```python
# python/tests/gui/test_your_gui.py
import pytest

def test_window_creation():
    """ウィンドウが正しく作成されることをテスト"""
    from PyQt6.QtWidgets import QMainWindow, QWidget
    
    window = QMainWindow()
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    
    # スタブ環境では例外が発生しないことを確認
    assert window is not None
    assert central_widget is not None
```

... (略後続セクションは元ファイルを参照)

````
