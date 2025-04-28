# api_server.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from run import generate_answer  # assumes both .py files are in same dir

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

    return GenerateResponse(
        answer=answer,
        used_context=[h["name"] for h in hits]
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
