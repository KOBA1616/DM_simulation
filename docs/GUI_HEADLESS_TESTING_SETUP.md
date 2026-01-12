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

#### ボタンクリックイベントのテスト

```python
def test_button_click():
    """ボタンクリックイベントの処理をテスト"""
    from PyQt6.QtWidgets import QPushButton
    from unittest.mock import MagicMock
    
    button = QPushButton("Click Me")
    handler = MagicMock()
    
    # シグナル接続（スタブでは connect は MagicMock）
    button.clicked.connect(handler)
    
    # イベント発火をシミュレート
    button.clicked.emit()
    
    # ハンドラーが呼ばれたことを確認
    handler.assert_called_once()
```

#### コンボボックスの選択テスト

```python
def test_combobox_selection():
    """コンボボックスの選択とイベント処理をテスト"""
    from PyQt6.QtWidgets import QComboBox
    from unittest.mock import MagicMock
    
    combo = QComboBox()
    combo.addItem("Option 1", 1)
    combo.addItem("Option 2", 2)
    
    handler = MagicMock()
    combo.currentIndexChanged.connect(handler)
    
    # インデックス変更
    combo.setCurrentIndex(1)
    combo.currentIndexChanged.emit(1)
    
    # ハンドラーが正しい引数で呼ばれたことを確認
    handler.assert_called_with(1)
```

#### エラーハンドリングのテスト

```python
def test_error_handling_in_handler():
    """イベントハンドラー内のエラー処理をテスト"""
    from PyQt6.QtWidgets import QPushButton
    
    button = QPushButton("Error Test")
    errors = []
    
    def faulty_handler():
        raise ValueError("Test error")
    
    def safe_handler():
        try:
            faulty_handler()
        except ValueError as e:
            errors.append(str(e))
    
    button.clicked.connect(safe_handler)
    button.clicked.emit()
    
    # エラーが正しくキャッチされた
    assert len(errors) == 1
    assert "Test error" in errors[0]
```

#### 複雑なウィジェット間の相互作用テスト

```python
def test_widget_state_management():
    """複数ウィジェット間の状態管理をテスト"""
    from PyQt6.QtWidgets import QPushButton, QComboBox
    
    # アプリケーションロジックをシミュレート
    class AppState:
        def __init__(self):
            self.mode = "default"
            self.count = 0
        
        def set_mode(self, index):
            self.mode = "mode_a" if index == 0 else "mode_b"
        
        def increment(self):
            self.count += 1
    
    state = AppState()
    
    mode_combo = QComboBox()
    mode_combo.addItem("Mode A", 0)
    mode_combo.addItem("Mode B", 1)
    
    action_button = QPushButton("Execute")
    
    # イベント接続
    mode_combo.currentIndexChanged.connect(state.set_mode)
    action_button.clicked.connect(state.increment)
    
    # 操作シミュレーション
    mode_combo.setCurrentIndex(1)
    mode_combo.currentIndexChanged.emit(1)
    assert state.mode == "mode_b"
    
    action_button.clicked.emit()
    assert state.count == 1
```

### スタブに新しいクラスを追加する場合

必要に応じて `run_pytest_with_pyqt_stub.py` の `setup_gui_stubs()` 関数にクラスを追加：

```python
# run_pytest_with_pyqt_stub.py の setup_gui_stubs() 内
for w in ['QLabel', 'QPushButton', ..., 'YourNewWidget']:
    setattr(qt_widgets, w, type(w, (DummyQWidget,), {}))
```

## CI/CD 統合

### GitHub Actions の例

```yaml
name: Tests with GUI Stubbing

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      
      - name: Verify GUI stubbing infrastructure
        run: |
          python run_pytest_with_pyqt_stub.py python/tests/gui/test_gui_stubbing.py -v
      
      - name: Run all tests (headless)
        run: |
          python run_pytest_with_pyqt_stub.py --cov=dm_toolkit --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### GitLab CI の例

```yaml
test:headless:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements-dev.txt
    - python run_pytest_with_pyqt_stub.py python/tests/gui/test_gui_stubbing.py -v
    - python run_pytest_with_pyqt_stub.py --cov=dm_toolkit
  coverage: '/TOTAL.*\s+(\d+%)$/'
```

## トラブルシューティング

### 問題: `ImportError: No module named 'PyQt6'`

**原因**: スタビングが正しく初期化されていない

**解決策**:
1. `run_pytest_with_pyqt_stub.py` を使用してテストを実行していることを確認
2. `test_gui_stubbing.py` を実行してスタビング機能が動作することを確認

### 問題: `AttributeError: module 'PyQt6.QtWidgets' has no attribute 'XXX'`

**原因**: スタブに必要なクラスが定義されていない

**解決策**:
`run_pytest_with_pyqt_stub.py` の `setup_gui_stubs()` に必要なクラスを追加：

```python
qt_widgets.YourMissingClass = type('YourMissingClass', (DummyQWidget,), {})
```

### 問題: テストが実際のGUIライブラリを使用してしまう

**原因**: 環境に PyQt6 がインストールされており、スタブより先に読み込まれている

**解決策**:
1. `conftest.py` の `_setup_minimal_gui_stubs()` が先に実行されることを確認
2. 必要に応じて実際の PyQt6 をアンインストール（開発環境のみ）

### 問題: CI環境でテストがタイムアウト

**原因**: GUI初期化がブロックしている可能性

**解決策**:
環境変数を設定：
```bash
export QT_QPA_PLATFORM=offscreen
export DISPLAY=:99  # Xvfb使用時
```

## ベストプラクティス

### 1. テストの独立性を保つ

```python
# Good: テストごとに新しいインスタンスを作成
def test_widget_a():
    widget = QWidget()
    # テスト...

def test_widget_b():
    widget = QWidget()  # 新しいインスタンス
    # テスト...

# Bad: グローバル状態を共有
global_widget = QWidget()  # ❌

def test_widget_a():
    global global_widget
    # テスト...
```

### 2. スタブの限界を理解する

スタブは実際のGUIイベントループやレンダリングを提供しません：

```python
# ✅ スタブで可能なこと
def test_structure():
    """ウィジェット構造とロジックのテスト"""
    window = QMainWindow()
    widget = QWidget()
    window.setCentralWidget(widget)
    assert window is not None

def test_event_logic():
    """イベントハンドラーのロジックテスト"""
    button = QPushButton()
    handler = MagicMock()
    button.clicked.connect(handler)
    button.clicked.emit()  # シグナルを明示的に発火
    handler.assert_called_once()

def test_state_changes():
    """ウィジェット状態の変更テスト"""
    combo = QComboBox()
    combo.addItem("Item 1")
    combo.setCurrentIndex(0)
    # ロジックの検証が可能

# ❌ スタブでは不可能なこと（実際のGUI環境が必要）
def test_visual_rendering():
    """視覚的なレンダリング検証"""
    window = QMainWindow()
    window.show()
    # ピクセル比較、スクリーンショット検証など ❌

def test_real_user_interaction():
    """実際のマウス/キーボード操作"""
    button = QPushButton()
    # QTest.mouseClick(button, Qt.LeftButton) ❌
    # 実際のクリックイベント生成は不可

def test_layout_geometry():
    """実際のレイアウト計算"""
    widget = QWidget()
    # widget.geometry() の実際の値取得 ❌
    # レイアウトエンジンは動作しない
```

**スタブ環境でのテスト戦略:**
- ✅ ビジネスロジックのテスト（状態管理、データ処理）
- ✅ イベントハンドラーの動作検証（emit で明示的に発火）
- ✅ ウィジェット構造の検証
- ✅ エラーハンドリングのテスト
- ❌ 視覚的な検証（別の統合テストで実施）
- ❌ 実際のユーザーインタラクション（E2Eテストで実施）

### 3. スタブ固有の動作をテストしない

```python
# Good: ビジネスロジックをテスト
def test_data_processing():
    processor = DataProcessor()
    result = processor.process(data)
    assert result == expected

# Bad: スタブの実装詳細に依存
def test_stub_behavior():
    widget = QWidget()
    assert isinstance(widget, DummyQWidget)  # ❌ スタブ依存
```

## まとめ

このヘッドレステスト環境により、以下が実現できます：

### ✅ テスト可能な操作

| 操作 | サポート | テスト方法 |
|------|---------|-----------|
| **ボタンクリック** | ✅ 完全 | `button.clicked.emit()` でシミュレート |
| **コンボボックス選択** | ✅ 完全 | `combo.currentIndexChanged.emit(index)` |
| **テキスト入力** | ✅ 完全 | `lineEdit.textChanged.emit(text)` |
| **チェックボックス** | ✅ 完全 | `checkbox.stateChanged.emit(state)` |
| **エラー検出** | ✅ 完全 | try/except でハンドラー内のエラーをキャッチ |
| **状態管理** | ✅ 完全 | ウィジェット間の相互作用と状態変更を検証 |
| **シグナル接続** | ✅ 完全 | `connect()`, `disconnect()` が実際に動作 |
| **レイアウト** | ⚠️ 部分的 | 構造検証のみ（実際の配置計算なし） |
| **視覚検証** | ❌ 不可 | スクリーンショット、ピクセル比較は別途必要 |
| **実UIイベント** | ❌ 不可 | QTest による実際のクリックは不可 |

### 🎯 利点

✅ CI/CD パイプラインでGUI関連のテストを自動実行  
✅ GUI環境がないサーバーでの開発・テスト  
✅ 高速なテスト実行（実際のGUI初期化が不要）  
✅ 98%+ のテストカバレッジを維持  
✅ **ボタン、コンボボックス、テキスト入力などの操作を完全にテスト可能**  
✅ **イベントハンドラー内のエラーを正確に検出**

### 📋 テスト例

```python
# ボタンクリックのテスト
def test_button_click():
    button = QPushButton("Click Me")
    handler = MagicMock()
    button.clicked.connect(handler)
    button.clicked.emit()  # クリックをシミュレート
    handler.assert_called_once()  # ✅ 呼び出しを検証

# コンボボックス選択のテスト
def test_combo_selection():
    combo = QComboBox()
    combo.addItem("Option 1", 1)
    handler = MagicMock()
    combo.currentIndexChanged.connect(handler)
    combo.setCurrentIndex(0)
    combo.currentIndexChanged.emit(0)  # 選択をシミュレート
    handler.assert_called_with(0)  # ✅ 正しい引数で呼ばれた

# エラー検出のテスト
def test_error_handling():
    button = QPushButton()
    errors = []
    def faulty_handler():
        raise ValueError("Error!")
    def safe_handler():
        try:
            faulty_handler()
        except ValueError as e:
            errors.append(str(e))
    button.clicked.connect(safe_handler)
    button.clicked.emit()
    assert len(errors) == 1  # ✅ エラーが検出された
```

**標準的な実行方法:**
```bash
# ヘッドレス環境でのテスト実行
python run_pytest_with_pyqt_stub.py

# スタビング検証
python run_pytest_with_pyqt_stub.py python/tests/gui/test_gui_stubbing.py -v

# インタラクションテストの実行
python run_pytest_with_pyqt_stub.py python/tests/gui/test_gui_interactions.py -v
```

詳細については、以下のファイルも参照してください：
- [AGENTS.md](../AGENTS.md) - 開発ポリシーとアーキテクチャガイドライン
- [run_pytest_with_pyqt_stub.py](../run_pytest_with_pyqt_stub.py) - メイン実行スクリプト
- [python/tests/conftest.py](../python/tests/conftest.py) - pytest設定とスタブセットアップ
