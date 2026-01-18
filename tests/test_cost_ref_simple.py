"""
Simple test to check if cost_ref attribute is available in FilterDef.
"""
import dm_ai_module as dm

def test_cost_ref_attribute():
    """Test that FilterDef has cost_ref attribute."""
    
    # Create a simple FilterDef
    filter_def = dm.FilterDef()
    
    # Check if cost_ref attribute exists
    try:
        # Try to access cost_ref
        val = filter_def.cost_ref
        print(f"✅ cost_ref attribute exists. Initial value: {val}")
        
        # Try to set it
        filter_def.cost_ref = "test_value"
        print(f"✅ cost_ref can be set. New value: {filter_def.cost_ref}")
        
        return True
    except AttributeError as e:
        print(f"❌ cost_ref attribute not found: {e}")
        
        # List available attributes
        print("\nAvailable attributes:")
        for attr in dir(filter_def):
            if not attr.startswith('_'):
                print(f"  - {attr}")
        
        return False

if __name__ == "__main__":
    try:
        result = test_cost_ref_attribute()
        if result:
            print("\n✅ Test passed!")
        else:
            print("\n❌ Test failed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
