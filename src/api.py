from functools import lru_cache
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from .core import Store, answer

app = FastAPI(title='Grounded RAG', version='1.0.0')

@lru_cache

def store():
    return Store()

class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)

@app.get('/health')
def health():
    return {'status': 'ok', 'chunks': store().count()}

@app.post('/documents')
async def upload(file: UploadFile = File(...)):
    try:
        payload = await file.read(20 * 1024 * 1024 + 1)
        return {'source': file.filename, 'chunks': store().ingest(file.filename or '', payload)}
    except (ValueError, Exception) as exc:
        if isinstance(exc, ValueError):
            raise HTTPException(400, str(exc)) from exc
        raise HTTPException(400, 'Could not extract this document') from exc

@app.post('/ask')
def ask(query: Query):
    hits = store().search(query.question, query.k)
    return {'answer': answer(query.question, hits), 'sources': hits}
