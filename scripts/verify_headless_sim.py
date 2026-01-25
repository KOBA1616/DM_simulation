# -*- coding: utf-8 -*-
"""
Verification script for Headless Simulation logic.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.batch_simulation import BatchSimulationRunner

def main():
    print("Initializing BatchSimulationRunner...")

    # Mock card_db (minimal)
    card_db = {}

    # We need a valid scenario name from SCENARIOS
    from dm_toolkit.training.scenario_definitions import SCENARIOS
    if not SCENARIOS:
        print("No scenarios found in SCENARIOS. Skipping run.")
        return

    scenario_name = list(SCENARIOS.keys())[0]

    runner = BatchSimulationRunner(
        card_db=card_db,
        scenario_name=scenario_name,
        episodes=2, # Short run
        threads=1,
        sims=10,
        evaluator_type="Random" # Fast
    )

    print(f"Running simulation on {scenario_name}...")

    def on_progress(p, m):
        # Only print major updates to avoid clutter
        if p % 50 == 0:
            print(f"[{p}%] {m}")

    win_rate, summary = runner.run(on_progress)

    print(f"Result: Win Rate {win_rate}%")
    print("Summary:")
    print(summary)

    print("Headless Simulation Logic Verification Passed!")

if __name__ == "__main__":
    main()
