import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dm_ai_module

print("python", sys.version)
print("dm_ai_module.__file__", getattr(dm_ai_module, "__file__", None))
print("dm_ai_module module type", type(dm_ai_module))

if hasattr(dm_ai_module, "PassiveType"):
    members = [n for n in dir(dm_ai_module.PassiveType) if ("CANNOT" in n or "FORCE" in n)]
    print("PassiveType new-ish members", members)
else:
    print("PassiveType missing")
