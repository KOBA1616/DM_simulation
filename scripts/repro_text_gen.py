
import sys
import os
import json

# Setup paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'python'))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator

def load_cards():
    with open('data/cards.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    cards = load_cards()
    target_ids = [1, 2, 3, 4, 6, 7]
    
    with open('repro_text_result.txt', 'w', encoding='utf-8') as outfile:
        outfile.write("-" * 50 + "\n")
        for card in cards:
            if card['id'] in target_ids:
                outfile.write(f"ID: {card['id']} Name: {card['name']}\n")
                try:
                    text = CardTextGenerator.generate_text(card)
                    outfile.write("Generated Text:\n")
                    outfile.write(text + "\n")
                except Exception as e:
                    outfile.write(f"Error generating text: {e}\n")
                outfile.write("-" * 50 + "\n")

if __name__ == "__main__":
    main()
