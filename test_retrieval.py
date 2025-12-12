from src.retriever import GraphRetriever
from src.models import Intent, Entities
from dotenv import load_dotenv

def test_retrieval():
    load_dotenv()
    print("Testing GraphRetriever...")
    retriever = GraphRetriever()
    
    # Test 1: Search Hotels in City
    print("\n[Test 1] Search hotels in Cairo...")
    intent = Intent(category="search", reasoning="test")
    entities = Entities(city="Cairo")
    results = retriever.retrieve_baseline(intent, entities)
    print(f"Found {len(results)} hotels.")
    if results: print(results[0])

    # Test 2: Filter by Rating
    print("\n[Test 2] Hotels with Rating >= 8.5...")
    intent = Intent(category="search", reasoning="test")
    entities = Entities(min_rating=8.5)
    results = retriever.retrieve_baseline(intent, entities)
    print(f"Found {len(results)} hotels.")
    
    # Test 3: Visa Check
    print("\n[Test 3] Visa from Egypt to France...")
    intent = Intent(category="search", reasoning="test")
    entities = Entities(current_country="Egypt", target_country="France")
    results = retriever.retrieve_baseline(intent, entities)
    print(f"Result: {results}")

    retriever.close()

if __name__ == "__main__":
    try:
        test_retrieval()
    except Exception as e:
        print(f"Test Failed: {e}")
