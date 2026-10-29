# Grounded RAG: cited document Q&A

Upload text PDFs or TXT documents, search them using a Python TF-IDF vector index, and generate grounded answers with an LLM. Each answer includes numbered references to retrieved passages; the UI shows document names, page numbers, text and similarity. SQLite persists chunks; the HNSW index is reconstructed from SQLite at startup and after replacing a file.

## Run

Requires Python 3.10+ and an OpenAI API key for generation. Retrieval and evaluation work without an API key. No C++ compiler or local model download is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key  # Windows PowerShell: $env:OPENAI_API_KEY="your_key"
uvicorn src.api:app --reload
```

In a second terminal: `streamlit run app.py`. API docs: http://localhost:8000/docs. Data is stored under `data/` by default. Never commit your API key or private documents.

## Architecture

PDF/TXT → page extraction → overlapping word chunks → Python TF-IDF vectors → exact cosine search → top-k context → LLM answer with numbered citations. The response includes the actual passages to inspect. Uploaded files are identified by filename; uploading the same name replaces its indexed content. Scanned PDFs require OCR before upload.

## API

```bash
curl -F 'file=@manual.pdf' http://localhost:8000/documents
curl -H 'Content-Type: application/json' -d '{"question":"What is the recommended voltage?","k":5}' http://localhost:8000/ask
```

## Evaluate

Create `questions.jsonl` with one JSON object per line: `{"question":"...","source":"manual.pdf","page":2}`. After indexing the same files run `python evaluate.py questions.jsonl --k 5`. This reports retrieval Recall@k, **not** answer accuracy. Test unanswerable questions manually and verify the answer declines to invent details. No performance metric is claimed until tested on a real labeled dataset.

## Limitations

LLM citations are requested by the prompt, but may be incorrect; inspect returned passages. No authentication, rate limiting, OCR, hybrid reranking, or production deployment is included. Do not expose this development server publicly.

## Suggested resume text (after running it)

Built a document Q&A system using Python TF-IDF vector search, FastAPI, and an LLM; returned source passages with page references and implemented a labeled Recall@k evaluation script.
