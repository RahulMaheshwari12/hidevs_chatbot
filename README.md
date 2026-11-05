# HiDevs RAG Chatbot

A document-grounded support agent for HiDevs. Answers are generated only from a single authoritative knowledge base (`data/knowledge_base.txt`), which is chunked, embedded with Gemini, and stored in a local Qdrant vector database.

The agent is part of the larger HiDevs website and is used directly from Python. There is no HTTP API in the current scope.

## How It Works

```
data/knowledge_base.txt
        │
        ▼  ingestion/        load → section-aware chunking
        ▼  embeddings/       Gemini embeddings (RETRIEVAL_DOCUMENT)
        ▼  vector_store/     upsert into Qdrant (local Docker container)
        │
User question
        ▼  agent/            model routes each message: knowledge_query, small_talk, or unclear
        ▼  retrieval/        Gemini query embedding → top-k Qdrant search
        ▼  prompts/          system prompt + retrieved context
        ▼  models/           Gemini chat model → grounded answer + sources
```

- **Routing is done by the model, not by keyword lists.** Each message gets one of three intents:
  - `knowledge_query`: a HiDevs question. It goes through retrieval and is answered only from the knowledge base.
  - `small_talk`: greetings, thanks, or goodbyes. The model replies directly, with no retrieval.
  - `unclear`: a message that lacks context, such as "what is that". The agent asks what the user means instead of guessing.
- If retrieval finds nothing relevant, the agent says the knowledge base does not contain the answer rather than guessing.
- The knowledge base is split by numbered section headings (e.g. `15.1 GenAI Internship`) and by `SOURCE n` blocks, so every chunk keeps its section and source metadata.

## Project Structure

```
HiDevs_RAG_Chatbot/
├── .env.example            # Template for environment variables
├── environment.yml         # Conda environment definition
├── requirements.txt        # pip dependencies
├── data/
│   └── knowledge_base.txt  # Single source of truth for the agent
├── ingestion/              # Load, chunk, and sync the knowledge base into Qdrant
│   ├── ingest.py           # ← first-run script
│   ├── text_loader.py
│   └── chunker.py
├── embeddings/
│   └── gemini_embeddings.py
├── vector_store/
│   ├── client.py           # Qdrant connection
│   └── collections.py      # Collection management
├── retrieval/
│   ├── vector_search.py    # Query embedding + Qdrant search
│   └── pipeline.py         # Retrieval + grounded answer generation
├── models/
│   ├── llm_provider.py     # Gemini chat model setup
│   └── chat.py             # Chat service wrapper
├── prompts/
│   ├── system_prompt.txt
│   └── templates.py        # System prompt and RAG prompt template
├── agent/
│   └── hidevs_agent.py     # HiDevsSupportAgent (entry point for the website)
├── tests/                  # Pytest and manual check scripts
├── utils/
├── docs/
├── app/                    # FastAPI service for the website frontend
│   ├── main.py             # App setup, CORS, agent startup
│   ├── api/routes.py       # /api/health and /api/chat
│   └── schemas/chat.py     # Request and response models
└── logs/
```

## Prerequisites

- **Python 3.11** (conda or venv)
- **Docker Desktop**, to run Qdrant locally
- **A Gemini API key** from Google AI Studio

## 1. Environment Setup

### Create the Python environment

Option A: Conda (recommended, matches `environment.yml`)

```bash
conda env create -f environment.yml
conda activate hidevs_rag
```

Option B: venv with pip

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / CMD
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### Create the `.env` file

Copy the template and fill in your values:

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

At minimum, set:

```env
GEMINI_API_KEY=your_actual_key
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=3072

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=hidevs_knowledge_base
```

Notes:

- `.env` is git-ignored. Never commit it or paste the key anywhere.
- `EMBEDDING_DIMENSION` must be `3072` for `gemini-embedding-001`. Ingestion stops with an error if the model returns a different size.
- Chunking is controlled by `CHUNK_SIZE` (default `1000`), `CHUNK_OVERLAP` (default `200`), and `RETRIEVAL_TOP_K` (default `5`).

### Start Qdrant

Run Qdrant in Docker, storing its data in the project's `qdrant_storage/` folder:

```bash
docker run -d --name hidevs-qdrant -p 6333:6333 -p 6334:6334 -v "%cd%/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

PowerShell users can use `-v "${PWD}/qdrant_storage:/qdrant/storage"`.

Check that it is running by opening <http://localhost:6333/dashboard> in a browser. If the container already exists from an earlier run, use `docker start hidevs-qdrant` instead.

## 2. First Run: Build the Vector Database

Run the ingestion script once after setup, and again any time `data/knowledge_base.txt` changes. **Run it from the project root**, because the knowledge base path is relative:

```bash
python -m ingestion.ingest
```

What it does:

1. Loads `data/knowledge_base.txt`
2. Splits it into section-aware chunks
3. Generates Gemini embeddings for every chunk
4. Creates the Qdrant collection if it does not exist
5. Uploads the vectors and metadata
6. Verifies the upload, then removes any outdated vectors

A successful run ends with a summary like this:

```
========== INGESTION COMPLETE ==========
Collection: hidevs_knowledge_base
Documents loaded: 1
Chunks generated: ...
Vectors uploaded: ...
Outdated vectors removed: ...
Verified vectors: ...
Qdrant points count: ...
=========================================
```

Re-running is safe. Point IDs are deterministic, so unchanged chunks are updated in place, and stale chunks are removed only after the new ones are verified.

If it fails, check that Docker/Qdrant is running and that `GEMINI_API_KEY` and `EMBEDDING_DIMENSION` are set correctly.

## 3. HTTP API (FastAPI)

The FastAPI service exposes the agent to the website frontend.

Start it from the project root:

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs are at <http://localhost:8000/docs>.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Returns `{"status": "ok"}` |
| POST | `/api/chat` | Send one message, get one reply |

Example request:

```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"message\": \"What is LeetPrompt?\"}"
```

Example response:

```json
{
  "answer": "LeetPrompt is a dedicated prompt-engineering practice platform...",
  "intent": "knowledge_query",
  "sources": [
    {
      "source_title": "...",
      "section_number": "...",
      "section_title": "LeetPrompt",
      "content": "..."
    }
  ]
}
```

- `intent` is one of `knowledge_query`, `small_talk`, `unclear`, or `off_topic`. Only `knowledge_query` replies include `sources`.
- Messages must be 1 to 2000 characters. Empty messages return `422`. Internal errors return a generic `500` message, and the details are logged on the server.
- The agent is created once at startup. The first start takes a few seconds while it connects to Qdrant.
- Browsers can call the API only from origins listed in `CORS_ORIGINS` (comma-separated, set in `.env`). The default allows `http://localhost:3000` and `http://localhost:5173`. Add your website's domain there before deploying.
- The API has no authentication yet. Add it before exposing the API publicly.

Run the API tests (no Gemini or Qdrant calls):

```bash
python -m pytest tests/test_api.py
```

## 4. Testing the Agent

Test the agent in this order, from the project root with the environment activated.

### Step 1: Check the connection and knowledge base (no API calls)

```bash
python -m tests.test_qdrant_client          # Qdrant reachable, lists collections
python -m pytest tests/test_ingestion.py    # Knowledge base loads and chunks correctly
```

### Step 2: Check retrieval quality

```bash
python -m tests.test_retrieval_quality
```

This runs a set of sample questions and prints the top matches with their scores and sections. Results should be the relevant HiDevs sections. Low scores or off-topic sections mean the knowledge base or chunking needs attention.

### Step 3: Chat with the agent in the terminal

```bash
python -m agent.hidevs_agent
```

Type a question and press Enter. Type `exit` or `quit` to stop. Each reply shows the intent and how many sources were retrieved.

Try questions like:

- `hi` or `how are you`, which should get a direct reply with no retrieval (`Sources retrieved: 0`)
- `what is that`, which should ask what you mean
- `What is CodeQuest and how does it work?`
- `What is LeetPrompt?`
- `What is the official customer support phone number for HiDevs?` This should be answered as not specified in the knowledge base, not invented.

### Step 4: Call the agent from Python (how the website uses it)

```python
from agent.hidevs_agent import HiDevsSupportAgent

agent = HiDevsSupportAgent()
response = agent.process_message("What is LeetPrompt?")

print(response["answer"])    # Generated answer
print(response["intent"])    # "knowledge_query", "small_talk" or "unclear"
print(response["sources"])   # Retrieved chunks with content and metadata
```

`process_message` raises `ValueError` for empty input. Other failures are logged and re-raised, so the website should wrap calls in its own error handling.

### Optional: Run the test suite

```bash
python -m pytest tests/
```

Some tests call Gemini and Qdrant, so the environment must be set up first.

## Configuration Reference

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Gemini API key |
| `GEMINI_CHAT_MODEL` | `gemini-3.5-flash-lite` | Model used for answers |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `GEMINI_TEMPERATURE` | `0.0` | Answer randomness (0.0 to 2.0) |
| `EMBEDDING_DIMENSION` | `3072` | Must match the embedding model output |
| `KNOWLEDGE_BASE_PATH` | `data/knowledge_base.txt` | Knowledge base file |
| `QDRANT_URL` | (empty) | Full hosted Qdrant URL, e.g. `https://xyz.cloud.qdrant.io:6333`. If set, it overrides host/port. |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Local Qdrant address (used when `QDRANT_URL` is empty) |
| `QDRANT_API_KEY` | (empty) | Only needed for a secured Qdrant |
| `QDRANT_COLLECTION_NAME` | `hidevs_knowledge_base` | Collection name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Chunking. Re-run ingestion after changing these. |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per question |
| `LOG_LEVEL` / `LOG_FILE` | `INFO` / `logs/app.log` | Logging |

## Updating the Knowledge Base

1. Edit `data/knowledge_base.txt`. Keep the `SOURCE n —` and numbered heading format so chunking works.
2. Run `python -m ingestion.ingest` again.
3. Check the answers with `python -m agent.hidevs_agent`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not connect to Qdrant` | Start the Docker container (`docker start hidevs-qdrant`) and check port 6333. |
| `Qdrant collection 'hidevs_knowledge_base' does not exist` | Run `python -m ingestion.ingest` first. |
| `GEMINI_API_KEY is missing` | Set the key in `.env` and run from the project root. |
| `Expected embedding dimension 3072, but received ...` | Set `EMBEDDING_DIMENSION` to the model's output size. |
| `Knowledge base file not found` | Run from the project root, or fix `KNOWLEDGE_BASE_PATH`. |
| Answers are outdated after editing the knowledge base | Re-run ingestion. The agent only reads what is in Qdrant. |

## Guardrails

- The agent answers only from retrieved HiDevs knowledge-base content. It does not guess prices, URLs, deadlines, or features.
- If the knowledge base does not cover a question, the agent says so.
- Internal implementation details (models, vector database, scores) are not exposed in answers.
