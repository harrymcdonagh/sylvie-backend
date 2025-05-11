import os
import logging
from typing import Optional
from openai import OpenAI
from kb import kb_instance
import tiktoken
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
api_key = os.getenv("OPENAI_API_KEY")


# Load OpenAI client using environment variable
client = OpenAI(api_key=api_key)

MAX_TOKENS = 10000
ENCODING = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    return len(ENCODING.encode(text))

def trim_history_by_tokens(history, max_tokens=MAX_TOKENS):
    total_tokens = 0
    trimmed = []
    for msg in reversed(history):  # Start from most recent
        msg_tokens = count_tokens(msg["content"])
        if total_tokens + msg_tokens > max_tokens:
            break
        trimmed.insert(0, msg)
        total_tokens += msg_tokens
    return trimmed

def generate_chat_response(prompt: str, student_name: Optional[str] = None, model: str = "gpt-4o-mini", history: Optional[list] = None) -> str:
    context = kb_instance.retrieve(prompt)

    system_prompt = (
        "You are Sylvie, a compassionate mental-health support chatbot for UEA students."
        " When responding you must:\n"
        "1. Greet the user" + (f", {student_name}" if student_name else "") + ".\n"
        "2. Reflect back the emotion you sense.\n"
        "3. Validate their feelings.\n"
        "4. Offer 1–2 coping strategies or tips.\n"
        "5. Recommend UEA services from the context below.\n"
        "6. For crisis/self-harm, direct to Nightline or emergency services.\n"
        "7. Invite further questions.\n\n"
        "Context:\n" + context
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        history = trim_history_by_tokens(history)
        logger.info(f"History trimmed sent to model:\n{history}")
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    logger.info(f"Final messages payload:\n{messages}")

    response = client.responses.create(model=model, input=messages)
    return response.output_text
