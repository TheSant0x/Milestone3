# HoRuS Travel Assistant

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-green.svg)
![License](https://img.shields.io/badge/License-Educational-orange.svg)

A smart travel recommendation system powered by Graph RAG (Retrieval-Augmented Generation). Combines Knowledge Graphs with AI to provide personalized hotel recommendations and answer travel-related questions.

---

##  Demo

https://github.com/user-attachments/assets/dc8098d2-96f9-4d9f-b03b-e337b12a7021

---

## What Does It Do?

This assistant helps you:
- **Find Hotels**: Search for hotels by city, name, or specific criteria
- **Get Recommendations**: Receive personalized suggestions based on traveler type, age, ratings, and preferences
- **Read Reviews**: Access real traveler reviews and ratings
- **Check Visa Requirements**: Find out if you need a visa to travel between countries
- **Smart Search**: Uses semantic understanding to find hotels even when you don't know exact names

## How It Works

The system combines three powerful technologies:

1. **Natural Language Processing**: Understands your questions in plain English
2. **Knowledge Graph (Neo4j)**: Stores relationships between hotels, cities, reviews, and travelers
3. **Vector Embeddings**: Finds similar hotels based on meaning, not just keywords

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Neo4j database (local or cloud)
- Hugging Face API token (free)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TheSant0x/horus-travel-assistant.git
   cd horus-travel-assistant
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment**:
   
   Create a `.env` file in the project root with these credentials:
   ```
   NEO4J_URI=neo4j://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password_here
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Create the Knowledge Graph**:
   
   This loads hotel, user, and review data into Neo4j:
   ```bash
   python Create_kg.py
   ```

5. **Add semantic search capabilities** (optional but recommended):
   ```bash
   python main.py --add-embeddings
   ```

## Usage

### Interactive Mode

Start a conversation with the assistant:
```bash
python main.py
```

Then ask questions like:
- "Find hotels in Paris"
- "Show me family-friendly hotels with good ratings"
- "Does the Hilton have a pool?"
- "Where can I travel without a visa from the US?"

### Single Query Mode

Get a quick answer:
```bash
python main.py --query "Best hotels in Cairo"
```

### Web Interface

Launch the Streamlit app for a visual interface:
```bash
streamlit run streamlit_app.py
```

## Example Queries

**Search for Hotels**:
- "Show me hotels in London"
- "Find the Marriott in Dubai"

**Get Recommendations**:
- "Suggest romantic hotels for couples"
- "Best hotels for business travelers"
- "Clean hotels with good facilities"

**Ask Questions**:
- "What are the reviews for Hotel X?"
- "Which hotels exceed expectations?"

**Check Visa Requirements**:
- "Do I need a visa to travel from USA to France?"

## Project Structure

```
📁 horus-travel-assistant/
├── 📄 main.py                 # Main application entry point
├── 📄 Create_kg.py            # Knowledge Graph setup script
├── 📄 streamlit_app.py        # Web interface
├── 📁 src/                    # Core modules
│   ├── processor.py           # Natural language understanding
│   ├── retriever.py           # Database queries
│   ├── embeddings.py          # Semantic search
│   ├── inference.py           # AI response generation
│   ├── models.py              # Data structures
│   └── logger.py              # Logging utilities
├── 📁 assets/                 # UI assets (logos, avatars)
├── 📄 requirements.txt        # Python dependencies
├── 📄 .env                    # Your credentials (create this)
└── 📄 README.md               # This file
```

## Data Files Required

The system expects these CSV files in the project root:
- `hotels.csv` - Hotel information
- `users.csv` - Traveler profiles  
- `reviews.csv` - Hotel reviews
- `visa.csv` - Visa requirements between countries

## Troubleshooting

**"HF_TOKEN not found"**: Make sure your `.env` file contains a valid Hugging Face token

**"NEO4J_PASSWORD not found"**: Check that Neo4j is running and credentials are correct in `.env`

**Slow responses**: The first query may be slow as models load. Subsequent queries are faster.

**No results found**: Try running `python main.py --add-embeddings` to enable semantic search

## Features

✅ Intent classification (question, recommendation, search, greeting)  
✅ Entity extraction (cities, hotels, traveler types, ratings)  
✅ Graph database queries for structured data  
✅ Vector similarity search for semantic matching  
✅ Natural language responses  
✅ Support for multiple query types  
✅ Web and command-line interfaces  

## Credits

Built with:
- [Neo4j](https://neo4j.com/) - Graph Database
- [LangChain](https://langchain.com/) - LLM Framework
- [Hugging Face](https://huggingface.co/) - AI Models
- [Sentence Transformers](https://www.sbert.net/) - Text Embeddings
- [Streamlit](https://streamlit.io/) - Web Interface

## Acknowledgments

Special thanks to Copilot & Antigravity.