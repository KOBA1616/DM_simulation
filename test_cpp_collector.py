import sys
import os
import time

# Add python/bin to path
sys.path.append(os.path.join(os.getcwd(), 'python/bin'))

import dm_ai_module

def test_data_collection():
    print("Loading cards...")
    # Load minimal cards
    dm_ai_module.JsonLoader.load_cards("data/cards.json")

    # We can't easily access the loaded map from python directly to pass to DataCollector
    # Wait, the bindings for JsonLoader just populate the C++ singleton registry or similar?
    # Actually, JsonLoader::load_cards populates CardRegistry in C++.
    # But GameInstance/HeuristicAgent needs a `std::map`.
    # Bindings usually need to provide a way to get the loaded DB.
    # But `JsonLoader.load_cards` returns a map? No, it returns void in bindings.
    # Wait, in `src/engine/card_system/json_loader.cpp`, `load_cards` returns a map?
    # Let's check `json_loader.cpp` and bindings.

    # In bindings:
    # .def_static("load_cards", &dm::engine::JsonLoader::load_cards);

    # Let's check return type.
    # If it returns a map, we can pass it.

    # Assuming it returns the map.
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")

    print(f"Loaded {len(card_db)} cards.")

    collector = dm_ai_module.DataCollector(card_db)

    print("Starting collection (10 episodes)...")
    start_time = time.time()
    batch = collector.collect_data_batch(10)
    end_time = time.time()

    print(f"Collection finished in {end_time - start_time:.4f} seconds.")
    print(f"Collected {len(batch.states)} samples.")

    if len(batch.states) > 0:
        print(f"State size: {len(batch.states[0])}")
        print(f"Policy size: {len(batch.policies[0])}")
        print(f"Values count: {len(batch.values)}")

if __name__ == "__main__":
    test_data_collection()
