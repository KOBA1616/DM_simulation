import subprocess
import sys

result = subprocess.run(
    ["cmake", "--build", "build-msvc", "--config", "Release", "--target", "dm_ai_module"],
    capture_output=True, text=True, encoding='utf-8', errors='replace'
)

with open("build_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"EXIT_CODE: {result.returncode}\n")
    
    # Find error lines
    errors = [l for l in result.stdout.split('\n') + result.stderr.split('\n')
              if ('error C' in l or 'error LNK' in l or 'Build FAILED' in l)
              and 'warning' not in l.lower()]
    
    if errors:
        f.write("ERRORS FOUND:\n")
        for e in errors[:30]:
            f.write(e[:200] + "\n")
    else:
        f.write("NO ERRORS FOUND\n")
    
    # Find succeeded/failed summary
    summary = [l for l in result.stdout.split('\n') + result.stderr.split('\n')
               if 'succeeded' in l.lower() or 'FAILED' in l]
    f.write("\nBUILD SUMMARY:\n")
    for l in summary[:5]:
        f.write(l[:200] + "\n")

print(f"Done. Exit code: {result.returncode}")
