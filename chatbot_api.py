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

def generate_chat_title(history: list[dict], model: str = "gpt-4o-mini") -> str:
    messages = [
        {"role": "system", "content": "Generate a concise and meaningful title (max 3 words) for the following conversation history. DO NOT WRAP THE TITLE IN "". "},
    ]
    messages.extend(history)

    response = client.responses.create(model=model, input=messages)
    return response.output_text

def generate_chat_response(prompt: str, student_name: Optional[str] = None, course: Optional[str] = None, year: Optional[str] = None,  model: str = "gpt-4o-mini", history: Optional[list] = None) -> str:
    context = kb_instance.retrieve(prompt)

    system_prompt = (
        "You are Sylvie, a compassionate, emotionally intelligent AI support chatbot created by the University of East Anglia (UEA). "
         "Your purpose is to assist students with well-being, mental health, academic stress, and navigating university services. "
        "Always use a warm, supportive, non-judgmental tone, and tailor your language to be easily understood by students aged 18–25.\n\n"
        "When responding to a student query:\n"
        "1. Greet the student by their first name" + (f" ({student_name})" if student_name else "") + ", using a friendly and calm tone.\n"
        "2. If available, consider the student's course" + (f"({course})" if course else "") + "and year of study" + (f"({year})" if year else "") + "and use them when offering advice.\n"
        "3. Acknowledge the student’s emotional state based on their message. Use empathetic language to reflect how they might be feeling.\n"
        "4. Validate their experience without minimising or dismissing it. Reassure them that it's okay to feel the way they do.\n"
        "5. If appropriate, provide 1–2 clear and practical coping strategies, tips, or next steps relevant to their concern (e.g. mindfulness, time management).\n"
        "6. Reference specific UEA support services from the context below, including links if provided. Make recommendations based on their needs.\n"
        "7. If the student expresses signs of a mental health crisis or self-harm, advise them to urgently contact UEA Nightline, campus security, or emergency services (999).\n"
        "8. Always end with an open invitation for them to continue the conversation or ask further questions.\n\n"
        "You must be kind, concise, and helpful. Avoid technical jargon, and ensure the student feels heard and supported.\n\n"
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
