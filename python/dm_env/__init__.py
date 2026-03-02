# python/dm_env/__init__.py
"""dm_env: DM Engine の共通 Python インターフェース。

GUI・CLI・AI パイプラインすべてが同一の CommandDef ブリッジを使用する。
PyQt6 不要（GUI 利用時は dm_toolkit.gui が別途 PyQt6 を import する）。
再発防止: このパッケージ内のどのファイルにも PyQt6 / PySide6 を
  import しないこと。GUI 層のみに GUI 依存を閉じ込める。
"""
from python.dm_env import builders, headless_runner, renderers, repl

__all__ = ["builders", "headless_runner", "renderers", "repl"]
