import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from src.models import Intent, Entities

# Initialize LLM (Ensure OPENAI_API_KEY is set in .env)
# Using a temperature of 0 for deterministic outputs
llm = ChatOpenAI(model="gpt-4o", temperature=0)

class Preprocessor:
    def __init__(self):
        self.intent_chain = self._build_intent_chain()
        self.entity_chain = self._build_entity_chain()

    def _build_intent_chain(self):
        system_prompt = """You are an expert intent classifier for a Travel Assistant.
        Analyze the user's query and classify it into one of the following categories:
        - question: The user is asking for a specific fact (e.g., "Does Hotel X have a pool?", "Where is Paris?").
        - recommendation: The user is asking for suggestions (e.g., "Suggest a romantic hotel", "Where should I stay?").
        - search: The user is searching for a specific entity entry (e.g., "Show me the Hilton", "Find user 123").
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        return prompt | llm.with_structured_output(Intent)

    def _build_entity_chain(self):
        system_prompt = """You are an expert Named Entity Recognizer (NER) for a Hotel Travel Assistant.
        Extract the following entities from the user's query:
        
        - Locations: Cities or Countries (e.g., "Paris", "France").
        - Hotels: Specific hotel names (e.g., "Hilton").
        - Traveller Types: e.g., Solo, Family, Couple, Business.
        - Attributes: Detailed constraints. Map common terms to these categories if possible:
            * "Clean" -> cleanliness
            * "Comfortable" -> comfort
            * "Good facilities/pool/wifi" -> facilities
            * "Central/Good location" -> location
            * "Cheap/Value" -> value_for_money
            * "Friendly staff" -> staff
        - Dates: specific dates, months, or duration.
        
        Only extract what is explicitly mentioned or strongly implied.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        return prompt | llm.with_structured_output(Entities)

    def process(self, query: str):
        print(f"Processing query: '{query}'")
        intent = self.intent_chain.invoke({"query": query})
        entities = self.entity_chain.invoke({"query": query})
        return intent, entities

if __name__ == "__main__":
    # Simple test if run directly
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment. Please set it in .env")
    else:
        processor = Preprocessor()
        queries = [
            "Find me a cheap hotel in Paris with a pool.",
            "Does the Hilton in London have good reviews?",
            "Suggest a place for a family trip to Italy."
        ]
        
        for q in queries:
            i, e = processor.process(q)
            print(f"\nQuery: {q}")
            print(f"Intent: {i.category} ({i.reasoning})")
            print(f"Entities: {e.dict()}")
