import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dm_ai_module

def test_heuristic():
    print("Testing HeuristicEvaluator...")
    try:
        card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
        evaluator = dm_ai_module.HeuristicEvaluator(card_db)
        print("Evaluator created.")
        
        gs = dm_ai_module.GameState(42)
        gs.setup_test_duel()
        dm_ai_module.PhaseManager.start_game(gs, card_db)
        
        # Test evaluate
        policies, values = evaluator.evaluate([gs])
        print(f"Evaluation result: Value={values[0]}, PolicySize={len(policies[0])}")
        
        # Test MCTS with heuristic
        mcts = dm_ai_module.MCTS(card_db)
        print("MCTS created.")
        
        policy = mcts.search_with_heuristic(gs, 10, evaluator, True, 1.0)
        print(f"Search result: PolicySize={len(policy)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_heuristic()
