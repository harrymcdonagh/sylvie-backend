# api_server.py
#uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from run import generate_answer

app = FastAPI()

class GenerateRequest(BaseModel):
    question: str
    top_k: int = 3

class GenerateResponse(BaseModel):
    answer: str
    used_context: List[str]

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    try:
        answer, hits = generate_answer(req.question, top_k=req.top_k)
        print("retrieved:", hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    used_context = [h["name"] for h in hits]

    sources_block = "\n\nHere are some relevant UEA resources:\n" + "\n".join(f" • {name}" for name in used_context)
    full_answer = answer + sources_block

    return GenerateResponse(
        answer=full_answer,
        used_context=used_context
    )

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
