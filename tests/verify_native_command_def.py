
import sys
import os

# Add bin/ to sys.path to find dm_ai_module
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
    print("dm_ai_module imported successfully.")
except ImportError as e:
    print(f"Failed to import dm_ai_module: {e}")
    sys.exit(1)

def verify_command_def():
    print("\n--- Verifying CommandDef ---")
    if not hasattr(dm_ai_module, 'CommandDef'):
        print("CommandDef NOT found in dm_ai_module.")
        return False

    cmd = dm_ai_module.CommandDef()
    print("CommandDef instantiated.")

    if hasattr(cmd, 'to_dict'):
        print("CommandDef has to_dict method.")

        # Test to_dict
        cmd.type = dm_ai_module.CommandType.DRAW_CARD
        cmd.amount = 2
        cmd.from_zone = "DECK"
        cmd.to_zone = "HAND"

        d = cmd.to_dict()
        print(f"to_dict result: {d}")

        if d.get('type') == dm_ai_module.CommandType.DRAW_CARD and d.get('amount') == 2:
            print("to_dict verification PASSED.")
        else:
            print("to_dict verification FAILED (values mismatch).")
            return False
    else:
        print("CommandDef does NOT have to_dict method.")
        return False
    return True

def verify_command_builder():
    print("\n--- Verifying CommandBuilder (native=True) ---")
    try:
        from dm_toolkit import command_builders
        # Mocking dm_ai_module in command_builders if it's not picked up automatically
        # (Though adding bin/ to sys.path should work if dm_toolkit imports it)

        # Reloading might be needed if it was imported before path update?
        # But this script runs fresh.

        cmd_obj = command_builders.build_draw_command(amount=3, native=True)
        print(f"Result type: {type(cmd_obj)}")

        if isinstance(cmd_obj, dm_ai_module.CommandDef):
            print("build_draw_command(native=True) returned CommandDef.")
            d = cmd_obj.to_dict()
            print(f"Converted to dict: {d}")
            if d.get('amount') == 3:
                 print("CommandBuilder verification PASSED.")
            else:
                 print("CommandBuilder verification FAILED (value mismatch).")
        else:
            print(f"build_draw_command(native=True) returned {type(cmd_obj)}, expected CommandDef.")
            return False

    except ImportError as e:
        print(f"ImportError testing command_builders: {e}")
        return False
    except Exception as e:
        print(f"Error testing command_builders: {e}")
        return False
    return True

if __name__ == "__main__":
    v1 = verify_command_def()
    v2 = verify_command_builder()

    if v1 and v2:
        print("\nOVERALL STATUS: SUCCESS")
    else:
        print("\nOVERALL STATUS: FAILURE")
        sys.exit(1)
