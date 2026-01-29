#!/usr/bin/env python3
"""Artifact manager: automatic cleanup, archiving and reporting for models and data.

Features:
- Prune by count and by total size for `models/` and `data/` directories.
- Archive old files into `archive/` (move or zip) when retention policies exceed limits.
- Produce JSON report with sizes, counts, and actions taken.
- Dry-run mode to preview actions.

Usage examples:
  python training/artifact_manager.py --dry-run
  python training/artifact_manager.py --max-model-bytes 2147483648 --max-data-bytes 524288000

Defaults conservative: keep recent 3 models, 5 data files; max model storage 2GB, data 500MB.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any


def human_size(n: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def list_files_sorted(dirpath: Path, pattern: str = '*') -> List[Path]:
    return sorted([p for p in dirpath.glob(pattern) if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)


def dir_total_size(files: List[Path]) -> int:
    return sum(p.stat().st_size for p in files)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def prune_by_count(files: List[Path], keep: int, dry_run: bool) -> List[Path]:
    removed = []
    for old in files[keep:]:
        removed.append(old)
        if not dry_run:
            try:
                old.unlink()
            except Exception:
                pass
    return removed


def prune_by_size(files: List[Path], max_bytes: int, dry_run: bool) -> List[Path]:
    removed = []
    total = dir_total_size(files)
    # files assumed sorted newest->oldest
    for p in reversed(files):  # remove oldest first
        if total <= max_bytes:
            break
        try:
            sz = p.stat().st_size
            removed.append(p)
            if not dry_run:
                p.unlink()
            total -= sz
        except Exception:
            continue
    return removed


def archive_files(files: List[Path], archive_dir: Path, zip_when: int = 10, dry_run: bool = True) -> List[Path]:
    """Move or zip files into archive_dir. If many files, create a zip bundle."""
    ensure_dir(archive_dir)
    moved = []
    if not files:
        return moved
    if len(files) >= zip_when:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        zip_path = archive_dir / f'archive_{timestamp}.zip'
        if not dry_run:
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    try:
                        zf.write(f, arcname=f.name)
                        f.unlink()
                        moved.append(zip_path)
                    except Exception:
                        continue
        else:
            moved.append(zip_path)
        return moved

    # else move individually
    for f in files:
        dst = archive_dir / f.name
        if not dry_run:
            try:
                shutil.move(str(f), str(dst))
                moved.append(dst)
            except Exception:
                continue
        else:
            moved.append(dst)
    return moved


def scan_and_manage(
    base_dir: Path,
    pattern: str,
    keep_count: int,
    max_bytes: int,
    archive_dir: Path,
    max_age_days: int,
    dry_run: bool,
) -> Dict[str, Any]:
    ensure_dir(base_dir)
    files = list_files_sorted(base_dir, pattern)
    total_before = dir_total_size(files)
    report = {
        'base_dir': str(base_dir),
        'pattern': pattern,
        'count_before': len(files),
        'size_before': total_before,
        'removed_by_count': [],
        'removed_by_size': [],
        'archived': [],
    }

    # prune by age first (archive old than max_age_days)
    to_archive_by_age = []
    if max_age_days > 0:
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        for f in files[::-1]:  # oldest first
            try:
                mtime = datetime.utcfromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    to_archive_by_age.append(f)
            except Exception:
                continue
    if to_archive_by_age:
        archived = archive_files(to_archive_by_age, archive_dir, dry_run=dry_run)
        report['archived'].extend([str(x) for x in archived])
        # refresh file list
        files = [f for f in files if f not in to_archive_by_age]

    # prune by count
    removed_cnt = prune_by_count(files, keep_count, dry_run)
    report['removed_by_count'] = [str(x) for x in removed_cnt]
    files = files[:keep_count]

    # prune by size
    files_all = list_files_sorted(base_dir, pattern)
    removed_sz = prune_by_size(files_all, max_bytes, dry_run)
    report['removed_by_size'] = [str(x) for x in removed_sz]

    report['count_after'] = len(list_files_sorted(base_dir, pattern))
    report['size_after'] = dir_total_size(list_files_sorted(base_dir, pattern))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--archive-dir', type=str, default='archive')
    parser.add_argument('--keep-models', type=int, default=3)
    parser.add_argument('--keep-data', type=int, default=5)
    parser.add_argument('--max-model-bytes', type=int, default=1 * 1024 * 1024 * 1024)
    parser.add_argument('--max-data-bytes', type=int, default=500 * 1024 * 1024)
    parser.add_argument('--max-model-age-days', type=int, default=90)
    parser.add_argument('--max-data-age-days', type=int, default=30)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report-out', type=str, default=None)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    archive_dir = Path(args.archive_dir)
    ensure_dir(archive_dir)

    overall = {'timestamp': datetime.utcnow().isoformat(), 'actions': []}

    # Manage models
    rep_models = scan_and_manage(
        models_dir,
        'duel_transformer_*.pth',
        args.keep_models,
        args.max_model_bytes,
        archive_dir / 'models',
        args.max_model_age_days,
        args.dry_run,
    )
    overall['actions'].append({'type': 'models', 'report': rep_models})

    # Manage onnx artifacts too
    rep_models_onnx = scan_and_manage(
        models_dir,
        '*.onnx',
        args.keep_models,
        args.max_model_bytes,
        archive_dir / 'models',
        args.max_model_age_days,
        args.dry_run,
    )
    overall['actions'].append({'type': 'models_onnx', 'report': rep_models_onnx})

    # Manage training data files
    rep_data = scan_and_manage(
        data_dir,
        'transformer_training_data_iter*.npz',
        args.keep_data,
        args.max_data_bytes,
        archive_dir / 'data',
        args.max_data_age_days,
        args.dry_run,
    )
    overall['actions'].append({'type': 'data', 'report': rep_data})

    # Save report
    if args.report_out:
        out_path = Path(args.report_out)
    else:
        out_dir = Path('reports')
        ensure_dir(out_dir)
        out_path = out_dir / f'artifact_report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'

    if not args.dry_run:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(overall, f, indent=2, ensure_ascii=False)
    else:
        # print summary to stdout
        print(json.dumps(overall, indent=2, ensure_ascii=False))

    print('Artifact management complete. Report:', out_path)


if __name__ == '__main__':
    main()


def run_artifact_manager(
    models_dir: str = 'models',
    data_dir: str = 'data',
    archive_dir: str = 'archive',
    keep_models: int = 3,
    keep_data: int = 5,
    max_model_bytes: int = 1 * 1024 * 1024 * 1024,
    max_data_bytes: int = 500 * 1024 * 1024,
    max_model_age_days: int = 90,
    max_data_age_days: int = 30,
    dry_run: bool = False,
    report_out: str | None = None,
):
    """Programmatic entrypoint for artifact management. Returns the report dict."""
    class Args:
        pass

    args = Args()
    args.models_dir = models_dir
    args.data_dir = data_dir
    args.archive_dir = archive_dir
    args.keep_models = keep_models
    args.keep_data = keep_data
    args.max_model_bytes = max_model_bytes
    args.max_data_bytes = max_data_bytes
    args.max_model_age_days = max_model_age_days
    args.max_data_age_days = max_data_age_days
    args.dry_run = dry_run
    args.report_out = report_out

    # reuse main logic by calling scan_and_manage
    models_dir_p = Path(args.models_dir)
    data_dir_p = Path(args.data_dir)
    archive_dir_p = Path(args.archive_dir)
    ensure_dir(archive_dir_p)

    overall = {'timestamp': datetime.utcnow().isoformat(), 'actions': []}

    rep_models = scan_and_manage(
        models_dir_p,
        'duel_transformer_*.pth',
        args.keep_models,
        args.max_model_bytes,
        archive_dir_p / 'models',
        args.max_model_age_days,
        args.dry_run,
    )
    overall['actions'].append({'type': 'models', 'report': rep_models})

    rep_models_onnx = scan_and_manage(
        models_dir_p,
        '*.onnx',
        args.keep_models,
        args.max_model_bytes,
        archive_dir_p / 'models',
        args.max_model_age_days,
        args.dry_run,
    )
    overall['actions'].append({'type': 'models_onnx', 'report': rep_models_onnx})

    rep_data = scan_and_manage(
        data_dir_p,
        'transformer_training_data_iter*.npz',
        args.keep_data,
        args.max_data_bytes,
        archive_dir_p / 'data',
        args.max_data_age_days,
        args.dry_run,
    )
    overall['actions'].append({'type': 'data', 'report': rep_data})

    # write report file if requested
    if args.report_out:
        out_path = Path(args.report_out)
    else:
        out_dir = Path('reports')
        ensure_dir(out_dir)
        out_path = out_dir / f'artifact_report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'

    if not args.dry_run:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(overall, f, indent=2, ensure_ascii=False)
    return overall
