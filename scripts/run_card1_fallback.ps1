# Run card1 tests in fallback mode
$env:DM_DISABLE_NATIVE = '1'
pytest tests/test_card1_hand_quality.py -q
