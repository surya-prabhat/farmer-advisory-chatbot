# Farmer Advisory Chatbot — Japfa Comfeed India

A RAG-based (Retrieval-Augmented Generation) conversational assistant that answers poultry and cattle farming queries in English and Hindi. Built as a technical assignment for Japfa Comfeed India, the chatbot grounds every response strictly in provided documents — no hallucination, no outside knowledge.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Python 3.x, LangChain 0.3.25 |
| LLM | Google Gemini 3.1 Flash (via API) |
| Embeddings | `nomic-embed-text` via Ollama | #Preferred over google or claude to negate rate limits
| Vector Store | ChromaDB |
| Keyword Search | BM25 (`rank_bm25`) |
| Translation | `deep-translator` + `langdetect` |
| UI | Streamlit |

---

## Features

- **Hybrid retrieval** — combines BM25 keyword search (60%) with ChromaDB semantic vector search (40%) using an EnsembleRetriever for higher recall across short and long queries
- **Multilingual** — detects Hindi or English input automatically, translates to English for retrieval, and returns the response in the original language
- **Grounded answers** — the LLM is instructed to use only the provided context; if the knowledge base doesn't contain relevant information, it says so explicitly
- **Four knowledge domains** — Disease Diagnostics, Disease Detection, Injection & Medication, Farming Insights

---

## How to Install and Run

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd farmer-advisory-chatbot

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the embedding model via Ollama (Make sure you have Ollama installed in your PC)
ollama pull nomic-embed-text

# 5. Add your Gemini API key
# Create a file named .env in the project root and add:
# GOOGLE_API_KEY=your_key_here

# 6. Run the app
streamlit run app.py
```

> The ChromaDB vector store is pre-built and committed to this repository. You do not need to run `ingest.py` unless you add new documents to the `docs/` folder.

---

## Known Limitations and How to Improve

### Knowledge base gaps

The current documents may lack information as they have been fetched as per requirements but do not cover some specific requirements if needed. When asked, the chatbot correctly responds that the information is not in its knowledge base. **To improve:** you can expand on the existing documents or add new/more structured documents to the `docs/` folder, then re-run `ingest.py`.
This is not a limitation from the app's end but just lack of information, adding new documents and rerunning `ingest.py` will fix the issue.

### Broadening the app's functionality

- **More languages** — `deep-translator` supports 100+ languages; adding regional languages like Telugu, Tamil, or Marathi requires no code changes, only testing
- **More knowledge domains** — drop additional `.txt` files into `docs/` and re-ingest; the retrieval pipeline handles new content automatically

