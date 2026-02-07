#!/usr/bin/env python3
"""C:\\tempディレクトリ内のファイルをチェック"""
import os
from pathlib import Path
from datetime import datetime

temp_dir = Path('C:/temp')

print("=" * 60)
print("C:\\temp Directory Check")
print("=" * 60)

if not temp_dir.exists():
    print(f"\n❌ Directory does NOT exist: {temp_dir}")
    print("\nCreating directory...")
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {temp_dir}")
else:
    print(f"\n✓ Directory exists: {temp_dir}")

print("\n" + "=" * 60)
print("Listing all .txt files in C:\\temp:")
print("=" * 60)

txt_files = list(temp_dir.glob('*.txt'))

if not txt_files:
    print("\n⚠️  No .txt files found in C:\\temp")
else:
    print(f"\nFound {len(txt_files)} .txt file(s):\n")
    for f in sorted(txt_files):
        size = f.stat().st_size
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        print(f"  📄 {f.name}")
        print(f"     Size: {size} bytes")
        print(f"     Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

print("=" * 60)
print("Checking for specific diagnostic files:")
print("=" * 60)

target_files = [
    'CRITICAL_RESOLVE_EFFECT_TEST.txt',
    'BEFORE_CPP_CALL.txt',
    'resolve_action_debug.txt',
]

for filename in target_files:
    filepath = temp_dir / filename
    if filepath.exists():
        print(f"\n✓ FOUND: {filename}")
        print(f"  Size: {filepath.stat().st_size} bytes")
        print("\n  === Content (first 500 chars) ===")
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            print(content[:500])
            if len(content) > 500:
                print(f"\n  ... (total {len(content)} characters)")
        except Exception as e:
            print(f"  Error reading file: {e}")
    else:
        print(f"\n❌ NOT FOUND: {filename}")

print("\n" + "=" * 60)
