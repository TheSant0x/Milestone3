import os
from dotenv import load_dotenv
from src.processor import Preprocessor

def main():
    load_dotenv()
    
    # checking for API Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[!] Error: OPENAI_API_KEY is missing.")
        print("Please open '.env' and add your API key to run this test.")
        return

    try:
        processor = Preprocessor()
        print("\n=== Graph-RAG Travel Assistant: Preprocessing Test ===")
        print("Type 'exit' to quit.\n")
        
        while True:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue

            print("...> Analyzing...")
            try:
                intent, entities = processor.process(user_input)
                
                print("-" * 40)
                print(f"Detected Intent:  [{intent.category.upper()}]")
                print(f"Reasoning:        {intent.reasoning}")
                print(f"Extracted Entities:")
                e_dict = entities.dict()
                for key, val in e_dict.items():
                    if val: # Only print non-empty
                        print(f"  - {key.replace('_', ' ').title()}: {val}")
                print("-" * 40 + "\n")
            except Exception as e:
                print(f"Error processing query: {e}")

    except Exception as e:
        print(f"Failed to initialize: {e}")

if __name__ == "__main__":
    main()
