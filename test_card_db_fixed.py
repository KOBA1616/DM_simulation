import sys
import dm_ai_module

print("=" * 60)
print("Testing card_db loading after JSON deserialization fixes")
print("=" * 60)

try:
    cdb = dm_ai_module.JsonLoader.load_cards('data/cards.json')
    print(f"\n✅ SUCCESS! Loaded {len(cdb)} cards\n")
    
    if len(cdb) > 0:
        print("First 5 cards loaded:")
        for i, card_id in enumerate(list(cdb.keys())[:5]):
            card = cdb[card_id]
            print(f"  {i+1}. ID={card.id}, Name={card.name}, Cost={card.cost}, Type={card.type}")
    
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ FAILED: {e}\n")
    sys.exit(1)
