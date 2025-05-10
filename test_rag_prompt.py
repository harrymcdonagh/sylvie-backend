#!/usr/bin/env python3
"""
test_rag_prompt.py

A prompt-only RAG test harness using your local model in ./model.

Dependencies:
    pip install sentence-transformers scikit-learn transformers torch

Usage:
    python test_rag_prompt.py ["Student question here"]
"""
import sys
import json
from pathlib import Path

# 1) Ensure retrieval dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("❌ Missing packages. Run: pip install sentence-transformers scikit-learn transformers torch", file=sys.stderr)
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 2) Load and embed UEA services KB
SERVICES_PATH = Path("uea_services.json")
if not SERVICES_PATH.exists():
    print(f"❌ Cannot find {SERVICES_PATH}", file=sys.stderr)
    sys.exit(1)
services = json.load(open(SERVICES_PATH, encoding="utf-8"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [svc.get("description", svc.get("desc", "")) for svc in services]
embs = embed_model.encode(texts, convert_to_numpy=True)

def retrieve_context(query: str, top_k: int = 3) -> str:
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embs)[0]
    ids = sims.argsort()[-top_k:][::-1]
    entries = []
    for i in ids:
        svc = services[i]
        name = svc.get("service", svc.get("name", "Service"))
        desc = svc.get("description", svc.get("desc", ""))
        url  = svc.get("url", "")
        text = f"- {name}: {desc}" + (f" ({url})" if url else "")
        entries.append(text)
    return "\n".join(entries)

# 3) Load tokenizer and base model (prompt-only)
BASE_MODEL_PATH = #choose model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
model     = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map="auto")
model.eval()

def build_prompt(context: str, question: str, name: str = None) -> str:
    # Greeting: include student name if provided
    greet = f"Hello, {name}!" if name else "Hello!"
    # Construct prompt with clear newlines
    prompt = f"""
System: You are Sylvie, a compassionate mental-health support chatbot for UEA students. Provide direct, complete answers. Do NOT ask follow-up or rhetorical questions. Your goals are to:
1. {greet}
2. Reflect back the emotion you sense ("I'm really sorry to hear you're feeling X…").
3. Validate it ("What you're going through makes total sense…").
4. Offer 1–2 concrete coping strategies or self-care tips.
5. Recommend 1–3 relevant UEA services from the context below.
6. If crisis or self-harm is mentioned, include: "If you ever feel like you might act on these thoughts, please reach out right now to Nightline (https://norwich.nightline.ac.uk/) or call 999/111 — you're not alone and help is here."
7. Invite further questions.

Context:
{context}

User: {question}
Assistant:""".strip()
    return prompt

if __name__ == "__main__":
    # Single interaction
    user_q = sys.argv[1] if len(sys.argv) > 1 else input("Student: ")
    context = retrieve_context(user_q)
    prompt = build_prompt(context, user_q)
    print("\n=== Sending Prompt ===\n", prompt, file=sys.stderr)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.75,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract assistant reply only
    reply = decoded.split("Assistant:")[-1].strip()
    print(f"Chatbot: {reply}")
