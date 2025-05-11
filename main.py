from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_api import generate_chat_response
from typing import Optional, List, Dict

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    student_name: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

@app.post("/api/generate")
async def generate(chat: ChatRequest):
    print(f"Request received:\nPrompt: {chat.prompt}\nStudent: {chat.student_name}\nHistory: {chat.history}")
    reply = generate_chat_response(chat.prompt, chat.student_name, history=chat.history)
    return {"reply": reply}
