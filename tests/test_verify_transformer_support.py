import sys
import os
import pytest
sys.path.append(os.path.abspath("bin"))

try:
    import dm_ai_module
except ImportError:
    pytest.fail("Could not import dm_ai_module. Ensure it is built and in bin/")

def test_neural_evaluator_transformer_mode():
    """Verify that NeuralEvaluator can be configured for Transformer mode and calls the sequence callback."""

    # 1. Setup GameState and Evaluator
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    if not card_db:
        # Fallback for environment where data might be missing, create dummy db
        # But we need real cards for tokens generally, though simple tokens might work.
        # Let's create a minimal card_db if load fails.
        pass

    evaluator = dm_ai_module.NeuralEvaluator(card_db)

    # 2. Configure for Transformer
    evaluator.set_model_type(dm_ai_module.ModelType.TRANSFORMER)

    # 3. Register Sequence Callback
    received_tokens = []

    def my_sequence_callback(batch_tokens):
        nonlocal received_tokens
        received_tokens = batch_tokens
        # Return dummy output
        batch_size = len(batch_tokens)
        policies = [[0.1] * 10] * batch_size
        values = [0.5] * batch_size
        return policies, values

    dm_ai_module.set_sequence_batch_callback(my_sequence_callback)

    # 4. Create dummy states and call evaluate
    states = []
    # Create a simple state (requires creating GameInstance to get initial state)
    # We can use setup_test_duel to populate it
    state = dm_ai_module.GameState(100) # dummy card count
    state.setup_test_duel()
    states.append(state)
    states.append(state) # batch of 2

    # Wrap states in list of shared_ptr (python binding handles this mostly as list of objects)
    # The binding for evaluate expects List[GameState] which are converted to vector<shared_ptr<GameState>>

    result = evaluator.evaluate(states)

    # 5. Verify
    assert len(received_tokens) == 2
    assert isinstance(received_tokens[0], list)
    assert len(received_tokens[0]) > 0
    assert isinstance(received_tokens[0][0], int)

    # Check output
    policies, values = result
    assert len(policies) == 2
    assert len(values) == 2

    print("Transformer mode verification passed.")

def test_neural_evaluator_resnet_mode_default():
    """Verify default ResNet mode uses flat callback or tensor conversion."""
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    evaluator = dm_ai_module.NeuralEvaluator(card_db)
    # Default is ResNet

    received_flat = []

    def my_flat_callback(flat, n, stride):
        nonlocal received_flat
        received_flat = flat
        policies = [[0.1] * 10] * n
        values = [0.5] * n
        return policies, values

    dm_ai_module.set_flat_batch_callback(my_flat_callback)
    dm_ai_module.clear_sequence_batch_callback() # Ensure we don't accidentally use it

    state = dm_ai_module.GameState(100)
    state.setup_test_duel()
    states = [state]

    evaluator.evaluate(states)

    assert len(received_flat) > 0
    assert isinstance(received_flat[0], float)

    print("ResNet mode verification passed.")

if __name__ == "__main__":
    test_neural_evaluator_transformer_mode()
    test_neural_evaluator_resnet_mode_default()
