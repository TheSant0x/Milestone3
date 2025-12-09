# Graph RAG Travel Assistant

This workspace contains the implementation of a Travel Assistant using Graph RAG (Retrieval-Augmented Generation). The system leverages a Knowledge Graph to provide context-aware travel recommendations.

## Project Structure

- **`main.py`**: Entry point for the application.
- **`Create_kg.py`**: Script to generate and populate the Knowledge Graph.
- **`src/`**: Source code directory containing data models (`models.py`) and core logic.
- **`config.txt`**: Configuration parameters for the application.
- **`requirements.txt`**: List of Python dependencies required for the project.
- **`.env`**: Environment variables (API keys, database credentials, etc.).

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
   Ensure your `.env` file is set up with the necessary credentials.

3. **Build Knowledge Graph**
   Run the creation script to initialize the knowledge base:
   ```bash
   python Create_kg.py
   ```

4. **Run the Assistant**
   Start the main application:
   ```bash
   python main.py
   ```
