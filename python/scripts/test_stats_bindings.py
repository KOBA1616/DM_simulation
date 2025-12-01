import dm_ai_module as dm

def main():
    # Load card DB
    card_db = dm.CsvLoader.load_cards("data/cards.csv")
    print(f"Loaded {len(card_db)} card definitions")

    # Create GameState and initialize stats
    gs = dm.GameState(12345)
    dm.initialize_card_stats(gs, card_db, 40)
    print("Initialized card stats map; sample sizes:")

    # Inspect a few card vectors
    for cid in list(card_db.keys())[:3]:
        vec = dm.vectorize_card_stats(gs, cid)
        print(f"card {cid} vec (len={len(vec)}):", vec)

    pot = dm.get_library_potential(gs)
    print("library potential (len=%d):" % len(pot), pot)

if __name__ == '__main__':
    main()
