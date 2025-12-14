
# Mock tr function
def tr(text):
    return text

# Logic extracted from CardDataManager._create_action_item
def generate_text(action):
    act_type = action.get('type', 'NONE')
    display_type = tr(act_type)

    if act_type == "MOVE_CARD":
             dest = action.get('destination_zone', 'HAND')
             display_type = tr(dest) # e.g. "Mana Charge", "Hand"

             # Contextual Naming
             if dest == "MANA_ZONE": display_type = "SEND_TO_MANA"
             elif dest == "GRAVEYARD": display_type = "DESTROY"
             elif dest == "HAND": display_type = "RETURN_TO_HAND"
             elif dest == "DECK_BOTTOM": display_type = "SEND_TO_DECK_BOTTOM"
             elif dest == "SHIELD_ZONE": display_type = "ADD_SHIELD"

             count = action.get('filter', {}).get('count')
             if count:
                 display_type += f" (Count: {count})"

    return f"Action: {display_type}"

def test_text_generation():
    # Test Mana Charge
    a1 = {"type": "MOVE_CARD", "destination_zone": "MANA_ZONE"}
    print(f"Test 1 (Mana): {generate_text(a1)}")
    assert "SEND_TO_MANA" in generate_text(a1)

    # Test Destroy
    a2 = {"type": "MOVE_CARD", "destination_zone": "GRAVEYARD"}
    print(f"Test 2 (Grave): {generate_text(a2)}")
    assert "DESTROY" in generate_text(a2)

    # Test Bounce with Count
    a3 = {"type": "MOVE_CARD", "destination_zone": "HAND", "filter": {"count": 2}}
    print(f"Test 3 (Bounce 2): {generate_text(a3)}")
    assert "RETURN_TO_HAND" in generate_text(a3)
    assert "Count: 2" in generate_text(a3)

if __name__ == "__main__":
    test_text_generation()
