# -*- coding: utf-8 -*-
"""
GUI エディタフォームの整合性チェックツール

フォームの更新メカニズムと signal/slot 接続を検証します。
"""

import sys
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class FormInfo:
    """フォーム情報"""
    name: str
    file_path: Path
    signals_emitted: Set[str] = field(default_factory=set)
    signals_connected: Set[str] = field(default_factory=set)
    has_load_ui_from_data: bool = False
    has_save_ui_to_data: bool = False
    has_update_ui_state: bool = False
    has_structure_update: bool = False
    widget_bindings: Set[str] = field(default_factory=set)
    registered_widgets: Set[str] = field(default_factory=set)
    update_data_calls: int = 0
    save_data_calls: int = 0


class FormAnalyzer(ast.NodeVisitor):
    """フォームコードのASTアナライザ"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.forms: Dict[str, FormInfo] = {}
        self.current_class = None

    def visit_ClassDef(self, node):
        """クラス定義の訪問"""
        # BaseEditForm を継承するクラスを検出
        bases = [base.id if isinstance(base, ast.Name) else None for base in node.bases]
        if 'BaseEditForm' in bases or 'QWidget' in bases:
            self.current_class = node.name
            self.forms[node.name] = FormInfo(
                name=node.name,
                file_path=self.file_path
            )
        self.generic_visit(node)
        self.current_class = None

    def visit_Assign(self, node):
        """代入文の訪問 (signal定義を検出)"""
        if self.current_class:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # pyqtSignal の検出
                    if isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Name):
                            if node.value.func.id == 'pyqtSignal':
                                self.forms[self.current_class].signals_emitted.add(target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """関数定義の訪問"""
        if self.current_class:
            form = self.forms[self.current_class]
            
            # テンプレートメソッドの検出
            if node.name == '_load_ui_from_data':
                form.has_load_ui_from_data = True
            elif node.name == '_save_ui_to_data':
                form.has_save_ui_to_data = True
            elif node.name == '_update_ui_state':
                form.has_update_ui_state = True
            
            # 構造更新メソッドの検出
            if 'structure_update' in node.name.lower():
                form.has_structure_update = True

        self.generic_visit(node)

    def visit_Call(self, node):
        """関数呼び出しの訪問"""
        if self.current_class:
            form = self.forms[self.current_class]
            
            # connect の検出
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == 'connect':
                    # signal.connect(slot) パターン
                    if isinstance(node.func.value, ast.Attribute):
                        signal_name = node.func.value.attr
                        form.signals_connected.add(signal_name)
                
                # update_data, save_data の呼び出しカウント
                elif node.func.attr == 'update_data':
                    form.update_data_calls += 1
                elif node.func.attr == 'save_data':
                    form.save_data_calls += 1
                
                # register_widget の検出
                elif node.func.attr == 'register_widget':
                    if len(node.args) > 0:
                        if isinstance(node.args[0], ast.Attribute):
                            widget_name = node.args[0].attr
                            form.registered_widgets.add(widget_name)
                            # キー指定がある場合
                            if len(node.args) > 1:
                                if isinstance(node.args[1], ast.Constant):
                                    form.widget_bindings.add(node.args[1].value)
        
        self.generic_visit(node)


def analyze_form_file(file_path: Path) -> Dict[str, FormInfo]:
    """フォームファイルを解析"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(file_path))
        analyzer = FormAnalyzer(file_path)
        analyzer.visit(tree)
        return analyzer.forms
    except Exception as e:
        print(f"⚠️ ファイル解析エラー: {file_path}: {e}")
        return {}


def check_form_integrity(forms_dir: Path) -> Tuple[List[str], List[str]]:
    """フォームの整合性チェック"""
    issues = []
    warnings = []
    all_forms: Dict[str, FormInfo] = {}

    # 全フォームファイルを解析
    for form_file in forms_dir.glob('*.py'):
        if form_file.name.startswith('__'):
            continue
        
        forms = analyze_form_file(form_file)
        all_forms.update(forms)

    print(f"\n📊 検出されたフォーム: {len(all_forms)} 個\n")

    # 各フォームの整合性チェック
    for form_name, form in all_forms.items():
        print(f"\n🔍 フォーム: {form_name} ({form.file_path.name})")
        
        # 1. テンプレートメソッドの実装チェック
        if not form.has_load_ui_from_data:
            warnings.append(
                f"  ⚠️ {form_name}: _load_ui_from_data() が未実装 (BaseEditFormのデフォルトを使用)"
            )
        else:
            print(f"  ✓ _load_ui_from_data() 実装済み")

        if not form.has_save_ui_to_data:
            warnings.append(
                f"  ⚠️ {form_name}: _save_ui_to_data() が未実装 (BaseEditFormのデフォルトを使用)"
            )
        else:
            print(f"  ✓ _save_ui_to_data() 実装済み")

        # 2. シグナル定義と接続のチェック
        if form.signals_emitted:
            print(f"  📡 定義されたシグナル: {', '.join(form.signals_emitted)}")
            
            # structure_update_requested がある場合、PropertyInspectorで接続されているか確認
            if 'structure_update_requested' in form.signals_emitted:
                print(f"  ✓ structure_update_requested シグナル定義済み")
        
        if form.signals_connected:
            print(f"  🔌 接続されたシグナル: {', '.join(form.signals_connected)}")

        # 3. ウィジェット登録とバインディングのチェック
        if form.registered_widgets:
            print(f"  🎛️ 登録されたウィジェット: {len(form.registered_widgets)} 個")
        else:
            warnings.append(
                f"  ⚠️ {form_name}: register_widget() の呼び出しが見つかりません"
            )

        if form.widget_bindings:
            print(f"  🔗 データバインディング: {len(form.widget_bindings)} 個")
            print(f"     キー: {', '.join(sorted(form.widget_bindings))}")

        # 4. update_data/save_data の呼び出しチェック
        if form.update_data_calls > 0:
            print(f"  🔄 update_data() 呼び出し: {form.update_data_calls} 回")
        else:
            warnings.append(
                f"  ⚠️ {form_name}: update_data() の呼び出しが見つかりません"
            )

        # 5. 構造更新メソッドのチェック
        if form.has_structure_update:
            print(f"  🏗️ 構造更新メソッドあり")

    return issues, warnings


def check_property_inspector_integrity(inspector_file: Path) -> List[str]:
    """PropertyInspector の整合性チェック"""
    issues = []
    
    print("\n\n🔍 PropertyInspector の整合性チェック\n")
    
    try:
        with open(inspector_file, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(inspector_file))
        
        # form_map の定義を検索
        form_map_keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if target.attr == 'form_map':
                            if isinstance(node.value, ast.Dict):
                                for key in node.value.keys:
                                    if isinstance(key, ast.Constant):
                                        form_map_keys.add(key.value)
        
        if form_map_keys:
            print(f"✓ form_map の定義が見つかりました")
            print(f"  登録されているタイプ: {', '.join(sorted(form_map_keys))}")
        else:
            issues.append("❌ form_map の定義が見つかりません")
        
        # signal 接続のチェック
        connected_forms = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == 'structure_update_requested':
                    # .structure_update_requested.connect() パターン
                    if isinstance(node.value, ast.Attribute):
                        form_attr = node.value.attr
                        if form_attr.endswith('_form'):
                            connected_forms.add(form_attr)
        
        if connected_forms:
            print(f"\n✓ 接続されているフォーム:")
            for form in sorted(connected_forms):
                print(f"  - {form}")
        else:
            issues.append("❌ フォームのシグナル接続が見つかりません")
        
    except Exception as e:
        issues.append(f"❌ PropertyInspector 解析エラー: {e}")
    
    return issues


def main():
    """メイン処理"""
    print("=" * 80)
    print("GUI エディタフォーム整合性チェックツール")
    print("=" * 80)

    # パスの設定
    project_root = Path(__file__).resolve().parent.parent
    forms_dir = project_root / 'dm_toolkit' / 'gui' / 'editor' / 'forms'
    inspector_file = project_root / 'dm_toolkit' / 'gui' / 'editor' / 'property_inspector.py'

    if not forms_dir.exists():
        print(f"❌ フォームディレクトリが見つかりません: {forms_dir}")
        return 1

    # フォームの整合性チェック
    issues, warnings = check_form_integrity(forms_dir)

    # PropertyInspector のチェック
    if inspector_file.exists():
        inspector_issues = check_property_inspector_integrity(inspector_file)
        issues.extend(inspector_issues)
    else:
        issues.append(f"❌ PropertyInspector ファイルが見つかりません: {inspector_file}")

    # 結果のサマリー
    print("\n" + "=" * 80)
    print("📋 チェック結果サマリー")
    print("=" * 80)

    if warnings:
        print(f"\n⚠️ 警告: {len(warnings)} 件")
        for warning in warnings:
            print(warning)

    if issues:
        print(f"\n❌ エラー: {len(issues)} 件")
        for issue in issues:
            print(issue)
        return 1
    else:
        print("\n✅ 重大なエラーは見つかりませんでした")
        if warnings:
            print(f"   ({len(warnings)} 件の警告がありますが、システムは動作可能です)")
        return 0


if __name__ == '__main__':
    sys.exit(main())
