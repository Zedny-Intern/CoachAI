# CoachAI (Multimodal Learning Coach)

CoachAI is a **Streamlit-based multimodal learning coach** that combines:

- A lesson/knowledge base (your own content)
- Retrieval (RAG) to ground answers in that knowledge
- An LLM backend (default: **Mistral API**) to generate explanations and practice content

The UI supports **text questions** and **image-based questions** (vision/OCR via the remote model).

## Functionality & features

- **Ask (Q&A)**
  - Ask questions with optional image upload.
  - Answers are grounded using retrieved lessons.
  - When applicable, responses cite retrieved document IDs.

- **Knowledge base (Lessons)**
  - Create, list, and delete lessons (topics).

- **RAG / Semantic search**
  - Searches relevant lessons using:
    - **pgvector** (preferred) when Postgres is configured
    - A fallback in-process similarity search when Postgres/pgvector is unavailable

- **Practice mode**
  - Generate practice questions from your saved lessons.
  - Submit answers and receive model feedback.

## How it works (runtime flow)

- **Streamlit entrypoint**: `app.py`
- **UI orchestration**: `coachai/ui/streamlit_utils.py` and `coachai/ui/learning_coach_agent.py`
  - The UI creates a `LearningCoachAgent`, which holds a `CoachService`.
- **Business logic**: `coachai/services/coach_service.py` (`CoachService`)
  - Retrieves lessons via `KnowledgeRepository`.
  - Calls the model backend via `ModelHandler` (default: remote Mistral API).
- **Retrieval & persistence**: `coachai/repositories/knowledge_repository.py` (`KnowledgeRepository`)
  - Loads lessons (from Supabase).

## Design & architecture

CoachAI follows a **layered architecture** with a clear dependency direction:

`ui (Streamlit)` -> `controllers` -> `services` -> `repositories` -> `client`

Key design choices:

- **Facade + internal modules (Option B naming)**
  - Public entrypoints stay stable (e.g. `CoachService`, `KnowledgeRepository`).
  - Implementation is split into internal modules (no underscore-prefixed filenames).

- **Side-effect-minimized package imports**
  - The `coachai` package is kept lightweight to reduce circular import risk.

### Repository structure (high-level)

- `app.py`
  - Streamlit entrypoint.
  - Imports UI code from `coachai/ui/*`.

- `coachai/ui/`
  - Streamlit UI layer: tabs, sidebar, session state, UI orchestration.

- `coachai/controllers/`
  - Transport adapters (UI/API) that translate inputs into service calls.

- `coachai/services/`
  - Business logic and orchestration (RAG + generation).
  - `coach_service.py` is the service facade.
  - `model_handler.py` contains model backend integration.

- `coachai/repositories/`
  - Data access and retrieval.
  - `knowledge_repository.py` is the repository facade.

- `coachai/client/`
  - External integrations (Supabase, Postgres, Cohere, Mistral).

- `coachai/api/`
  - FastAPI application and schemas.
  - Includes “protected” endpoints intended for trusted backends.

## Tech stack

- **UI**: Streamlit
- **Optional API**: FastAPI
- **LLM / Vision**: Mistral API (remote)
- **Auth + persistence + attachments**: Supabase
- **Vector search**: Postgres + pgvector
- **Embeddings**: Cohere
- **Config**: `python-dotenv` + environment variables

Python dependencies are declared in `requirements.txt`.

## Quick start

### 1) Create a virtual environment

```bash
python -m venv .coach_env
source .coach_env/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env` file at the repository root.

Minimal configuration (remote model):

```bash
MISTRAL_API_KEY=...
USE_REMOTE_MODEL=true
```

Optional (recommended) persistence + vector search:

```bash
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_DB_URL=...
```

### 4) Run the Streamlit app

Run from the repository root:

```bash
streamlit run app.py
```

You should then see URLs like:

- `Local URL: http://localhost:8501`

## Configuration reference (.env)

```bash
# --- Required (default remote-model mode) ---
MISTRAL_API_KEY=...

# --- Optional model settings ---
USE_REMOTE_MODEL=true
MISTRAL_API_URL=https://api.mistral.ai
MODEL_NAME=mistral-medium-2508
MISTRAL_OCR_MODEL=mistral-ocr-latest

# --- RAG tuning ---
TOP_K=3
TEMPERATURE=0.7
MAX_TOKENS=1024
TOP_P=0.9
REPETITION_PENALTY=1.05
LENGTH_PENALTY=1.0

# --- Supabase ---
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_DB_URL=
SUPABASE_STORAGE_BUCKET=attachments

# --- pgvector ---
PGVECTOR_DIMENSION=384

# --- Cohere embeddings ---
COHERE_API_KEY=
COHERE_MODEL=embed-multilingual-light-v3.0

# --- Optional server-side RAG endpoints ---
USE_SERVER_SIDE_RAG=false
```

## Optional: run the FastAPI server

The FastAPI app lives in `coachai/api/main.py`.

```bash
PYTHONPATH=. python coachai/api/main.py
```

Key endpoints:

- `GET /health`
- `POST /api/v1/search/`
- `POST /api/v1/protected/*` (requires `x-service-key` header matching `SUPABASE_SERVICE_ROLE_KEY`)