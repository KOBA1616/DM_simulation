"""モデルフォルダの不要ファイルをアーカイブ/削除するユーティリティ

使い方例:
  python scripts/cleanup_models.py --models-dir models --keep 2 --dry-run
  python scripts/cleanup_models.py --models-dir models --keep 2 --no-dry-run
  python scripts/cleanup_models.py --models-dir models --delete --confirm
  python scripts/cleanup_models.py --models-dir models --report

デフォルトでは `models/cleanup_config.json` があれば読み込みます。
再発防止: dry_run はデフォルト True。設定ファイルか --no-dry-run で明示的に無効化する。
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


def report_dir_size(label: str, directory: Path) -> None:
    """ディレクトリ内ファイルの総サイズをログ出力する。"""
    if not directory.exists():
        return
    total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    logger.info("%s size: %.1f MB", label, total / 1_048_576)


def cleanup_subdir(
    subdir: Path,
    keep: int,
    extensions: List[str],
    archive_dir: Path,
    dry_run: bool,
    do_delete: bool,
    confirm: bool,
    keep_total: bool = True,
) -> None:
    """サブディレクトリ (例: checkpoints/) 内の古いモデルを整理する。
    keep_total=True の場合、グループ化せず全ファイルを mtime 降順で並べ、
    新しい keep 件を残して残りをアーカイブ/削除する（checkpoints 向け）。
    再発防止: checkpoints/ に訓練中間ファイルが累積しやすいため設定で件数を制限する。
    """
    if not subdir.exists():
        return
    files = find_model_files(subdir, model_extensions_from_list(extensions))
    logger.info("Subdir %s: found %d files", subdir, len(files))
    if not files:
        return

    if keep_total:
        # 全ファイルを mtime 降順で並べ、新しい keep 件を超えた分を削除対象にする
        files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        to_remove = files_sorted[keep:]
    else:
        groups = group_by_base(files)
        moves_tuples = plan_cleanup(groups, keep)
        to_remove = [src for src, _ in moves_tuples]

    if not to_remove:
        logger.info("Subdir %s: nothing to archive (keep=%d)", subdir, keep)
        return
    logger.info("Subdir %s: %d files to archive/delete", subdir, len(to_remove))
    if do_delete:
        if not confirm:
            logger.error("--delete requires --confirm. Skipping subdir %s.", subdir)
            return
        for src in to_remove:
            if dry_run:
                logger.info("DRY RUN: would delete %s", src)
            else:
                logger.info("Deleting %s", src)
                src.unlink()
    else:
        dst_dir = archive_dir / subdir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in to_remove:
            dst = dst_dir / src.name
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
    dr = p.add_mutually_exclusive_group()
    dr.add_argument("--dry-run", dest="dry_run", action="store_true", default=None, help="only show actions, do not move files")
    dr.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="disable dry-run (actually move/delete files)")
    p.add_argument("--delete", action="store_true", help="delete instead of archiving (irreversible, requires --confirm)")
    p.add_argument("--confirm", action="store_true", help="confirm destructive actions like --delete")
    p.add_argument("--skip-checkpoints", action="store_true", help="skip cleanup of checkpoints/ subdir")
    p.add_argument("--report", action="store_true", help="print directory size before/after cleanup")
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
    exts = args.extensions if args.extensions is not None else cfg.get("extensions", ["pth", "pt", "onnx", "bin", "h5", "ckpt"])  # type: ignore
    # 再発防止: args.dry_run は True/False/None (--dry-run/--no-dry-run/未指定)。
    # 未指定のときは設定ファイルの dry_run を尊重する。
    if args.dry_run is None:
        dry_run = bool(cfg.get("dry_run", True))
    else:
        dry_run = args.dry_run

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.report:
        report_dir_size("Before", models_dir)

    files = find_model_files(models_dir, model_extensions_from_list(exts))
    logger.info("Found %d model files in %s", len(files), models_dir)

    groups = group_by_base(files)
    logger.info("Identified %d model groups", len(groups))

    moves = plan_cleanup(groups, keep)
    if moves:
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
    else:
        logger.info("No files to archive/delete in root (keep=%d)", keep)

    # --- checkpoints サブディレクトリの整理 ---
    if not args.skip_checkpoints:
        ckpt_cfg = cfg.get("checkpoints", {})
        ckpt_keep = int(ckpt_cfg.get("keep", keep))
        ckpt_exts = ckpt_cfg.get("extensions", ["pth", "pt", "ckpt"])
        ckpt_subdir = ckpt_cfg.get("subdir", "checkpoints")
        cleanup_subdir(
            subdir=models_dir / ckpt_subdir,
            keep=ckpt_keep,
            extensions=ckpt_exts,
            archive_dir=archive_dir,
            dry_run=dry_run,
            do_delete=args.delete,
            confirm=args.confirm,
            keep_total=bool(ckpt_cfg.get("keep_total", True)),
        )

    if args.report:
        report_dir_size("After", models_dir)


if __name__ == "__main__":
    main()
