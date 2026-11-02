# Conversational Agent with Memory

A Streamlit personal research assistant that ingests documents, keeps short-term conversation memory, compresses old turns into durable memory, and retrieves relevant context from both document and conversation memory.

## Requirements
- Python 3.11+
- Docker + Docker Compose (for containerized runs)
- Optional for live LLM responses: `OPENROUTER_API_KEY`

## Local Development
Install dependencies:

```bash
uv sync
```

Run tests:

```bash
pytest tests/ -v
```

Run app directly:

```bash
uv run streamlit run app.py
```

## Docker Profiles
Run app profile:

```bash
make app
```

Run debug profile (VS Code attach on port 5678, live bind mount):

```bash
make debug
```

Stop containers:

```bash
make down
```

Only run one profile at a time (`app` or `debug`) so both can use port `8501`.

## OpenRouter
The app reads `OPENROUTER_API_KEY` from your local environment at runtime.

- Model: `openai/gpt-4.1-mini`
- Endpoint: OpenRouter OpenAI-compatible API

Do not commit keys to the repository.
