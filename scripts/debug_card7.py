import json
import pprint

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def main() -> None:
    with open("data/cards.json", "r", encoding="utf-8") as f:
        cards = json.load(f)

    card = next(c for c in cards if c.get("id") == 7)
    print("name:", card.get("name"))
    print("--- body text ---")
    print(CardTextGenerator.generate_body_text(card))

    print("--- commands (flattened) ---")
    flattened = []
    for eff in card.get("effects", []) or []:
        for cmd in eff.get("commands", []) or []:
            flattened.append(cmd)
            if isinstance(cmd, dict) and "options" in cmd:
                for chain in cmd.get("options", []) or []:
                    for nested in chain:
                        flattened.append(nested)

    for cmd in flattened:
        if isinstance(cmd, dict):
            ctype = cmd.get("type") or cmd.get("name")
            if ctype in {"APPLY_MODIFIER", "ADD_KEYWORD", "MUTATE", "GRANT_KEYWORD"}:
                pprint.pp(cmd)


if __name__ == "__main__":
    main()
