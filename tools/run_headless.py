#!/usr/bin/env python
# tools/run_headless.py
"""DM Engine — ヘッドレス実行エントリポイント

使い方:
  python tools/run_headless.py --mode ai-vs-ai
  python tools/run_headless.py --mode human-vs-ai
  python tools/run_headless.py --mode batch --games 100 --output result.json

再発防止: このファイルに PyQt6 / PySide6 の import を追加しないこと。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（python パッケージを解決するため）
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.dm_env.headless_runner import run_game
from python.dm_env.repl import run_interactive


def main() -> None:
    parser = argparse.ArgumentParser(description="DM Engine Headless Runner")
    parser.add_argument(
        "--mode", choices=["ai-vs-ai", "human-vs-ai", "batch"], default="ai-vs-ai",
        help="実行モード (default: ai-vs-ai)"
    )
    parser.add_argument("--games", type=int, default=1, help="バッチ試合数 (default: 1)")
    parser.add_argument("--output", default=None, help="結果 JSON 出力先パス")
    parser.add_argument("--verbose", action="store_true", help="ターンログを出力する")
    args = parser.parse_args()

    if args.mode == "ai-vs-ai":
        result = run_game([], [], output_json=args.output, verbose=args.verbose)
        # winner が GameResult enum の場合 str 変換
        printable = {k: (str(v) if not isinstance(v, (int, str, list, type(None))) else v)
                     for k, v in result.items()}
        print(json.dumps(printable, ensure_ascii=False, indent=2))

    elif args.mode == "human-vs-ai":
        run_interactive([], [])

    elif args.mode == "batch":
        results = [run_game([], []) for _ in range(args.games)]
        winners = [r.get("winner") for r in results]
        # winner は GameResult enum / int / None など多様なため文字列化して集計
        w_strs = [str(w) for w in winners]
        summary = {
            "games": args.games,
            "p0_wins": sum(1 for w in w_strs if w in ("0", "GameResult.P0_WIN", "P0_WIN")),
            "p1_wins": sum(1 for w in w_strs if w in ("1", "GameResult.P1_WIN", "P1_WIN")),
            "draws_or_unresolved": sum(
                1 for w in w_strs
                if w not in ("0", "GameResult.P0_WIN", "P0_WIN",
                             "1", "GameResult.P1_WIN", "P1_WIN")
            ),
        }
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        print(
            f"P0: {summary['p0_wins']}勝  "
            f"P1: {summary['p1_wins']}勝  "
            f"未決: {summary['draws_or_unresolved']}  "
            f"({args.games}試合)"
        )


if __name__ == "__main__":
    main()
