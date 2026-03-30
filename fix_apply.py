import re

for path in ["python/tests/unit/test_serializer_apply_cir.py", "python/tests/unit/test_unified_condition_preview.py"]:
    with open(path, "r") as f:
        c = f.read()
    c = c.replace("*** End Patch", "")
    with open(path, "w") as f:
        f.write(c)
