import dm_ai_module as dm

pi = dm.PlayerIntent.RESOLVE_EFFECT
print(f"value={int(pi)}")
print(f"str={str(pi)}")
print(f"repr={repr(pi)}")
print(f"hasattr_name={hasattr(pi, 'name')}")
print(f"type={type(pi).__name__}")

# Try to get name if it exists
if hasattr(pi, 'name'):
    print(f"name={pi.name}")
else:
    print("No 'name' attribute")
