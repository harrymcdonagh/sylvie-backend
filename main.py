from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_api import generate_chat_response, generate_chat_title
from typing import Optional, List, Dict

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    student_name: Optional[str] = None
    course: Optional[str] = None
    year: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

@app.post("/api/generate")
async def generate(chat: ChatRequest):
    print(f"Request received:\nPrompt: {chat.prompt}\nStudent: {chat.student_name}\nCourse: {chat.course}\nYear: {chat.year}History: {chat.history}")
    reply = generate_chat_response(chat.prompt, chat.student_name, chat.course, chat.year, history=chat.history)
    title = generate_chat_title(chat.history + [{"role": "user", "content": chat.prompt}, {"role": "assistant", "content": reply}])
    return {"reply": reply, "title": title}
