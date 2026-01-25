# -*- coding: utf-8 -*-
"""
Verification script for Headless Editor logic.
This script demonstrates manipulating card data without PyQt6 dependencies
by using the VirtualStandardItemModel abstraction.
"""
import sys
import os
import json
import traceback

# Ensure dm_toolkit is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force headless mode if PyQt6 is present (to verify logic handles it)
# We can't easily unload modules, but we can verify imports
try:
    from dm_toolkit.gui.editor.data_manager import CardDataManager, QStandardItemModel, QModelIndex
    from dm_toolkit.gui.editor.virtual_model import VirtualStandardItemModel
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def main():
    print("Initializing CardDataManager in headless mode...")

    # Check if we are using Virtual or Real model
    if QStandardItemModel is VirtualStandardItemModel:
        print("Confirmed: Using VirtualStandardItemModel.")
    else:
        print("Note: Using Real QStandardItemModel (PyQt6 is available).")
        # To strictly test VirtualModel when PyQt6 is available, we'd need to mock import
        # or inject it manually. For now, we assume the logic works if tests pass.
        # But we can manually inject for this test!
        from dm_toolkit.gui.editor.virtual_model import VirtualStandardItemModel as VModel
        model = VModel()
        manager = CardDataManager(model)
        print("Forced injection of VirtualStandardItemModel for testing.")

    # Setup dummy data
    model = VirtualStandardItemModel()
    manager = CardDataManager(model) # DataManager handles the injection?
    # DataManager constructor expects QStandardItemModel type hint, but runtime is dynamic

    initial_data = [
        {
            "id": 1000,
            "name": "Headless Dragon",
            "type": "CREATURE",
            "civilizations": ["FIRE"],
            "cost": 5,
            "power": 5000,
            "effects": []
        }
    ]

    print("Loading data...")
    manager.load_data(initial_data)

    # Verify load
    full_data = manager.get_full_data()
    assert len(full_data) == 1
    assert full_data[0]['name'] == "Headless Dragon"
    print("Data loaded successfully.")

    # Add a new card
    print("Adding new card...")
    manager.add_new_card() # Adds "New Card" with generated ID

    full_data = manager.get_full_data()
    assert len(full_data) == 2
    new_card = full_data[1]
    print(f"New card added: {new_card['id']} - {new_card['name']}")

    # Find item for the new card to add effect
    root = model.invisibleRootItem()
    # Assume 2nd child is the new card (0-based index 1)
    new_card_item = root.child(1)

    # Add Effect
    print("Adding effect to new card...")
    # Manager doesn't expose add_effect directly on item, FeatureService does?
    # manager.feature_service.add_effect...
    # But usually UI calls manager.serializer methods or feature_service
    # Let's use the method similar to what UI uses

    # We can use serializer to add child
    eff_data = manager.create_default_trigger_data()
    manager.add_child_item(new_card_item, "EFFECT", eff_data, "Effect: ON_PLAY")

    full_data = manager.get_full_data()
    assert len(full_data[1]['effects']) == 1
    print("Effect added successfully.")

    print("Headless Editor Verification Passed!")

if __name__ == "__main__":
    main()
