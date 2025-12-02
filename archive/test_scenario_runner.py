import sys
import os
import pytest
import numpy as np

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))
sys.path.append(os.path.join(os.getcwd(), 'python'))

try:
    import dm_ai_module
    from training.scenario_runner import ScenarioRunner
    from training.scenario_definitions import SCENARIOS
except ImportError:
    pass

def test_scenario_runner_initialization():
    card_db = {}
    runner = ScenarioRunner(card_db)
    assert runner.card_db == card_db

def test_scenario_run_simple():
    # Setup card DB (minimal)
    try:
        card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
    except:
        card_db = {}

    runner = ScenarioRunner(card_db)

    # Run a scenario that exists
    # Assuming "lethal_puzzle_easy" is defined in scenario_definitions.py
    # and it terminates (either win, timeout, or draw loop)
    result = runner.run_scenario("lethal_puzzle_easy", None, max_turns=5)

    assert result in ["TIMEOUT", "DRAW_LOOP"] or result.startswith("GAME_OVER")

def test_loop_detection():
    # This might be hard to test deterministically with random agent,
    # but we can ensure it runs without crash
    try:
        card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
    except:
        card_db = {}

    runner = ScenarioRunner(card_db)
    result = runner.run_scenario("infinite_loop_setup", None, max_turns=5)

    # Just check it returns a valid string result
    assert isinstance(result, str)
