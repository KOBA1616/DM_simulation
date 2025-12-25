import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict


class Analytics:
    def __init__(self, data_dir="data/generations"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_generations(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, "gen_*.json")))
        generations = []
        for f in files:
            with open(f, 'r', encoding='utf-8') as fp:
                generations.append(json.load(fp))
        return generations

    def generate_heatmap(self, output_path="data/analytics/heatmap.png"):
        generations = self.load_generations()
        if not generations:
            print("No generation data found.")
            return

        # Collect all unique card IDs
        all_card_ids = set()
        for gen in generations:
            for deck in gen['decks']:
                all_card_ids.update(deck)

        sorted_card_ids = sorted(list(all_card_ids))
        card_map = {cid: i for i, cid in enumerate(sorted_card_ids)}

        # Create matrix: rows=cards, cols=generations
        matrix = np.zeros((len(sorted_card_ids), len(generations)))

        for col, gen in enumerate(generations):
            total_decks = len(gen['decks'])
            if total_decks == 0:
                continue

                counts: Dict[int, int] = {}
            for deck in gen['decks']:
                for cid in deck:
                    counts[cid] = counts.get(cid, 0) + 1

            for cid, count in counts.items():
                # Usage rate (avg copies per deck or % of decks containing it?)
                # Let's use avg copies per deck (0-4)
                matrix[card_map[cid], col] = count / total_decks

        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(
            matrix, aspect='auto', cmap='viridis', interpolation='nearest'
        )
        plt.colorbar(label='Avg Copies per Deck')
        plt.xlabel('Generation')
        plt.ylabel('Card ID')
        plt.title('Meta-Game Evolution: Card Usage Heatmap')

        # Y-ticks (Card IDs)
        if len(sorted_card_ids) < 50:
            plt.yticks(range(len(sorted_card_ids)), sorted_card_ids)
        else:
            plt.yticks(
                range(0, len(sorted_card_ids), 5), sorted_card_ids[::5]
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Heatmap saved to {output_path}")

    def generate_report(self, output_path="data/analytics/report.html"):
        generations = self.load_generations()
        if not generations:
            with open(output_path, 'w') as f:
                f.write("<h1>No Data</h1>")
            return

        html = ["<html><head><title>DM AI Analytics</title></head><body>"]
        html.append("<h1>Meta-Game Evolution Report</h1>")
        html.append(f"<p>Generated at: {datetime.now()}</p>")
        html.append(f"<p>Total Generations: {len(generations)}</p>")

        # Embed Heatmap
        self.generate_heatmap("data/analytics/heatmap.png")
        html.append(
            '<img src="heatmap.png" alt="Card Usage Heatmap" '
            'style="max-width:100%">'
        )

        html.append("<h2>Top Cards per Generation</h2>")

        for i, gen in enumerate(generations):
            html.append(f"<h3>Generation {i}</h3>")
            counts: Dict[int, int] = {}
            for deck in gen['decks']:
                for cid in deck:
                    counts[cid] = counts.get(cid, 0) + 1

            # Sort by count
            sorted_counts = sorted(
                counts.items(), key=lambda x: x[1], reverse=True
            )

            html.append("<ul>")
            for cid, count in sorted_counts[:5]:
                html.append(f"<li>Card {cid}: {count} copies total</li>")
            html.append("</ul>")

        html.append("</body></html>")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html))
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    # Generate dummy data if none exists
    if not glob.glob("data/generations/gen_*.json"):
        print("Generating dummy data...")
        os.makedirs("data/generations", exist_ok=True)
        for i in range(5):
            data = {
                "generation": i,
                "decks": [
                    [np.random.randint(1, 20) for _ in range(40)]
                    for _ in range(10)
                ]
            }
            with open(f"data/generations/gen_{i:03d}.json", 'w') as f:
                json.dump(data, f)

    analytics = Analytics()
    analytics.generate_report()
