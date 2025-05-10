import sys, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import torch

# 2) Load UEA services KB
SERVICES_PATH = Path("uea_services.json")
if not SERVICES_PATH.exists():
    print(f"❌ Cannot find {SERVICES_PATH}", file=sys.stderr)
    sys.exit(1)
services_kb = json.load(open(SERVICES_PATH, encoding="utf-8"))

# 3) Precompute embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [svc.get("description", svc.get("desc","")) for svc in services_kb]
embs  = embed_model.encode(texts, convert_to_numpy=True)

def retrieve_context(query: str, top_k: int = 3) -> str:
    q_emb    = embed_model.encode([query], convert_to_numpy=True)
    sims     = cosine_similarity(q_emb, embs)[0]
    top_ids  = sims.argsort()[-top_k:][::-1]
    lines    = []
    for idx in top_ids:
        svc  = services_kb[idx]
        name = svc.get("service", svc.get("name","Service"))
        desc = svc.get("description", svc.get("desc",""))
        url  = svc.get("url","")
        lines.append(f"- **{name}**: {desc}" + (f" ({url})" if url else ""))
    return "\n".join(lines)

# 4) Load quantized + offloaded base model + LoRA adapters
BASE_MODEL_PATH   = "./model"
LORA_ADAPTER_PATH = "./sylvie-lora"
OFFLOAD_DIR       = "offload"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
model.eval()

# 5) Clear out any leftover sampling settings
model.generation_config.temperature = 1.0
model.generation_config.top_p       = 1.0

# 6) Few-shot primer
FEW_SHOT = (
    "<|system|>\n"
    "You are Sylvie, an empathetic UEA student assistant. Provide complete, polite, and accurate answers.\n"
    "<|user|>\n"
    "Where can I find mental health support on campus?\n"
    "<|assistant|>\n"
    "You can contact UEA’s Student Wellbeing Service at wellbeing@uea.ac.uk or visit the Wellbeing Hub in The Hive.\n\n"
    "<|user|>\n"
    "How do I book an academic appeal?\n"
    "<|assistant|>\n"
    "Submit your appeal form via the Student Information Zone in The Hive or email academic.appeals@uea.ac.uk.\n\n"
    "<|user|>\n"
    "What services are available for career advice?\n"
    "<|assistant|>\n"
    "You can visit the UEA CareerCentral for career advice. They offer workshops, one-on-one sessions, and job listings. "
    "Check their website for more details."
)

# 7) RAG-enabled chat function
SYSTEM_TEMPLATE = (
    "Here are relevant UEA services for grounding:\n"
    "{context}\n\n"
    "Now answer the student’s question clearly and kindly."
)

def chat_with_rag(user_input: str) -> str:
    # 1. Retrieve context
    context = retrieve_context(user_input, top_k=3)

    # 2. Build system message
    system_msg = SYSTEM_TEMPLATE.format(context=context)

    # 3. Full prompt
    prompt = (
        FEW_SHOT +
        f"<|system|>\n{system_msg}\n"
        f"<|user|>\n{user_input}\n"
        f"<|assistant|>\n"
    )

    # Debugging
    print("\n=== FULL PROMPT ===\n", prompt, file=sys.stderr)

    # 4. Tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,   # allow rich, multi-sentence replies
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n=== RAW DECODED ===\n", decoded, file=sys.stderr)

    # 5. Extract assistant answer
    if "<|assistant|>" in decoded:
        reply = decoded.split("<|assistant|>")[-1]
        for tag in ("<|user|>", "<|system|>"):
            if tag in reply:
                reply = reply.split(tag)[0]
        return reply.strip()

    return (
        "I’m sorry, I’m having trouble answering right now. "
        "Please contact UEA Student Services for assistance."
    )

# 8) CLI entrypoint
if __name__ == "__main__":
    print("Sylvie (RAG+few-shot) is ready. Type 'exit' to quit.\n")
    while True:
        q = input("Student: ")
        if q.strip().lower() in {"exit","quit"}:
            print("Chatbot: Take care! 👋")
            break
        resp = chat_with_rag(q)
        print("Chatbot:", resp, "\n")
