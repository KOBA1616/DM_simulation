"""モデルフォルダの不要ファイルをアーカイブ/削除するユーティリティ

使い方例:
  python scripts/cleanup_models.py --models-dir models --keep 3 --dry-run

デフォルトでは `models/cleanup_config.json` があれば読み込みます。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


logger = logging.getLogger("cleanup_models")


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read config %s: %s", config_path, e)
        return {}


def model_extensions_from_list(exts: Iterable[str]) -> List[str]:
    return [e.lstrip(".") for e in exts]


def grouping_key(name: str) -> str:
    # heuristics: remove trailing version / timestamp tokens
    # examples removed: _v1, -20260129, .final, _checkpoint123
    s = re.sub(r"[\._-](?:v?\d+|20\d{6}|\d{8}T?\d+|final|best|checkpoint.*)$", "", name, flags=re.I)
    return s


def find_model_files(models_dir: Path, extensions: List[str]) -> List[Path]:
    exts = set(e.lower() for e in extensions)
    results: List[Path] = []
    for p in models_dir.iterdir():
        if p.is_file():
            if p.suffix.lstrip(".").lower() in exts:
                results.append(p)
    return results


def group_by_base(files: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in files:
        key = grouping_key(p.stem)
        groups.setdefault(key, []).append(p)
    return groups


def plan_cleanup(groups: Dict[str, List[Path]], keep: int) -> List[Tuple[Path, Path]]:
    # return list of (src, dst) moves for files to archive
    moves: List[Tuple[Path, Path]] = []
    for base, files in groups.items():
        # sort by mtime desc (newest first)
        files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        to_remove = files_sorted[keep:]
        for f in to_remove:
            moves.append((f, Path(base) / f.name))
    return moves


def apply_moves(moves: List[Tuple[Path, Path]], archive_dir: Path, dry_run: bool) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for src, rel in moves:
        dst = archive_dir / rel.name
        if dry_run:
            logger.info("DRY RUN: would move %s -> %s", src, dst)
        else:
            logger.info("Moving %s -> %s", src, dst)
            shutil.move(str(src), str(dst))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cleanup old model files in a models directory")
    p.add_argument("--models-dir", default="models", help="models directory")
    p.add_argument("--config", default=None, help="optional JSON config file path")
    p.add_argument("--keep", type=int, default=None, help="how many recent files to keep per model")
    p.add_argument("--archive-dir", default=None, help="where to move old files (default: models/archive)")
    p.add_argument("--extensions", nargs="*", default=None, help="extensions to consider (e.g. onnx pth)")
    p.add_argument("--dry-run", action="store_true", help="only show actions, do not move files")
    p.add_argument("--delete", action="store_true", help="delete instead of archiving (irreversible, requires --confirm)")
    p.add_argument("--confirm", action="store_true", help="confirm destructive actions like --delete")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        logger.error("models dir not found: %s", models_dir)
        return

    config_path = Path(args.config) if args.config else models_dir / "cleanup_config.json"
    cfg = load_config(config_path)

    keep = args.keep if args.keep is not None else int(cfg.get("keep", 3))
    archive_dir = Path(args.archive_dir) if args.archive_dir else Path(cfg.get("archive_dir", str(models_dir / "archive")))
    exts = args.extensions if args.extensions is not None else cfg.get("extensions", ["pth", "pt", "onnx", "bin", "h5", "ckpt", "data", "json"])  # type: ignore
    dry_run = args.dry_run or bool(cfg.get("dry_run", True))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    files = find_model_files(models_dir, model_extensions_from_list(exts))
    logger.info("Found %d model files in %s", len(files), models_dir)

    groups = group_by_base(files)
    logger.info("Identified %d model groups", len(groups))

    moves = plan_cleanup(groups, keep)
    if not moves:
        logger.info("No files to archive/delete (keep=%d)", keep)
        return

    logger.info("Planned %d files to archive/delete", len(moves))

    if args.delete:
        if not args.confirm:
            logger.error("Destructive action requested (--delete) but --confirm missing. Aborting.")
            return
        for src, _ in moves:
            if dry_run:
                logger.info("DRY RUN: would delete %s", src)
            else:
                logger.info("Deleting %s", src)
                src.unlink()
    else:
        apply_moves(moves, archive_dir, dry_run)


if __name__ == "__main__":
    main()
