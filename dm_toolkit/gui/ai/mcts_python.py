# -*- coding: utf-8 -*-
# 再発防止: 純粋Python MCTS実装は保守コストが高くバグが多かったため削除。
#   dm_ai_module.MCTS (C++) への委譲ラッパーに置き換え済み。
#   フォールバックとして HeuristicAgent を使用し、さらに失敗した場合はランダム選択。
#   MctsView や可視化ツールに渡すツリーデータは get_last_root() 経由で取得する。

from __future__ import annotations
from typing import Any, Callable, List, Optional

try:
    import dm_ai_module as _dm
except ImportError:
    _dm = None  # type: ignore[assignment]


class PythonMCTS:
    """dm_ai_module.MCTS (C++) への委譲ラッパー。

    search() は以下の優先順位でアクションを返す:
      1. C++ MCTS + HeuristicEvaluator による探索（dm_ai_module.MCTS が利用可能な場合）
      2. C++ HeuristicAgent によるヒューリスティック選択
      3. ランダム選択

    get_tree_data() は dm_ai_module.MCTS.get_last_root() 経由でツリーデータを返す。

    再発防止: should_stop は search() のシグネチャ互換のために保持するが、
      C++ MCTS 側では使用しない（即時実行）。
    """

    def __init__(
        self,
        card_db: Any = None,
        simulations: int = 100,
        c_puct: float = 1.0,
    ) -> None:
        self.card_db = card_db
        self.simulations = simulations
        self.c_puct = c_puct
        self.should_stop: Optional[Callable[[], bool]] = None
        self._native: Any = None
        self._last_root: Any = None

        if _dm is not None and card_db is not None:
            try:
                # dm_ai_module.MCTS(card_db, c_puct, dirichlet_alpha, dirichlet_epsilon, batch_size, alpha)
                self._native = _dm.MCTS(card_db, c_puct, 0.3, 0.25)
            except Exception:
                self._native = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, root_state: Any) -> Any:
        """最善手を探索して CommandDef (またはそれに相当するオブジェクト) を返す。"""
        if _dm is None:
            return None

        legal_cmds = self._get_legal_commands(root_state)

        # 1. C++ MCTS による探索（シンプルなランダム評価器を使用）
        if self._native is not None and legal_cmds:
            try:
                n_actions = len(legal_cmds)

                def _uniform_evaluator(states: List[Any]):
                    # 均一な事前分布と中立的な価値推定を返す
                    priors = [[1.0 / n_actions] * n_actions for _ in states]
                    values = [0.5] * len(states)
                    return priors, values

                policy: List[float] = self._native.search(
                    root_state, self.simulations, _uniform_evaluator,
                    False, 1.0,
                )
                self._last_root = self._native.get_last_root()
                if policy and len(policy) >= len(legal_cmds):
                    best_idx = max(range(len(legal_cmds)), key=lambda i: policy[i])
                    return legal_cmds[best_idx]
            except Exception:
                pass  # フォールバックへ

        # 2. HeuristicAgent フォールバック
        if legal_cmds and self.card_db is not None:
            try:
                player_id: int = getattr(root_state, "active_player_id", 0)
                agent = _dm.HeuristicAgent(player_id, self.card_db)
                result = agent.get_command(root_state, legal_cmds)
                if result is not None:
                    return result
            except Exception:
                pass

        # 3. ランダム選択
        if legal_cmds:
            import random
            return random.choice(legal_cmds)
        return None

    def get_tree_data(self) -> dict:
        """MCTS ツリーデータを可視化用の辞書形式で返す。"""
        if self._last_root is None:
            return {}
        try:
            return self._node_to_dict(self._last_root)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_legal_commands(self, state: Any) -> List[Any]:
        from dm_toolkit import commands as cmds_module  # 遅延インポートで循環参照回避
        try:
            result = cmds_module.generate_legal_commands(
                state, self.card_db, strict=False, skip_wrapper=True
            ) or []
        except TypeError:
            try:
                result = cmds_module.generate_legal_commands(state, self.card_db) or []
            except Exception:
                result = []
        except Exception:
            result = []
        return result

    def _node_to_dict(self, node: Any) -> dict:
        if node is None:
            return {}
        try:
            action = getattr(node, "action", None)
            name = str(action) if action else "Root"
            visits = int(getattr(node, "visit_count", 0))
            value = float(getattr(node, "value", 0.0))
            children_raw: List[Any] = list(getattr(node, "children", []))
            children = sorted(
                [self._node_to_dict(c) for c in children_raw],
                key=lambda d: d.get("visits", 0),
                reverse=True,
            )
            return {"name": name, "visits": visits, "value": value, "children": children}
        except Exception:
            return {}

