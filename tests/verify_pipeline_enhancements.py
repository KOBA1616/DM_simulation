
import sys
import os

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))

try:
    import dm_ai_module
except ImportError as e:
    print(f"Error importing dm_ai_module: {e}")
    sys.exit(1)

def test_pipeline_load():
    # Setup state
    try:
        state = dm_ai_module.GameState(10)
        print("GameState created successfully")
    except Exception as e:
        print(f"Failed to create GameState: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline_load()
