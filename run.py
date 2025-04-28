# model_runner.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from kb_query import get_top_services
from typing import List, Tuple, Dict

MODEL_DIR = "sylvie-finetuned"

# Load tokenizer & model once
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")


def generate_answer(question: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
    """
    Retrieves top_k KB entries, builds prompt, runs the model,
    and returns (answer, used_entries).
    """
    # 1) KB lookup
    hits = get_top_services(question, k=top_k)

    # 2) Build prompt inline with human-like style instruction
    context = "\n".join(
        f"- {h['name']}: {h['desc']} ({h['url']})"
        for h in hits
    )
    prompt = f"""
You are Sylvie, the friendly UEA student support assistant who chats like a trusted friend. When you reply, always do the following:

1.Start with a warm, personal greeting, using a casual, approachable tone that feels human, not scripted. You can comment lightly on the feeling or mood you sense from the user's message.

2.Weave in exactly the services listed below — using their real names naturally — as if you’re casually recommending something helpful over coffee with a friend. Make it feel personal, not like reading a brochure.
3.Stay conversational: elaborate on your advice, share "insider tips" (e.g., “a lot of students say xxx service is super friendly!”), little relatable comments, or examples where possible. Never list services like a menu or dump links.
4.Stick only to the services provided in {context} — do not make up new services or suggest outside resources.
5.Never tell the user to use another chatbot. Always invite them to keep talking to you for more support, making it clear you’re happy to keep chatting or helping them through things.
6.Always reflect the user's emotion and validate it gently — make them feel seen and understood before offering any suggestions.
7.Focus on connection over solutions. Prioritize emotional support and friendly conversation even if the user seems to just want to vent.
8.Keep replies a good balance of supportive and practical: be empathetic but also helpful where appropriate.
9.Keep answers moderately detailed (3–5 sentences usually), but if the student sounds very distressed, keep your reply shorter, softer, and offer small next steps.
10.Never assume urgency — if the user says something serious (like mental health crises), encourage them kindly to reach out to professional UEA services mentioned in {context} without sounding alarming.

Here are the services you can draw on:
{context}

Student question:
{question}

Your answer: """.strip()

    # 3) Tokenize & generate with repetition controls
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 4) Decode only the newly generated tokens
    gen_tokens = output_ids[0][ inputs["input_ids"].shape[-1]: ]
    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return answer, hits


if __name__ == "__main__":
    print("Sylvie local REPL. Ctrl+C to exit.")
    try:
        while True:
            q = input("\n> ")
            ans, used = generate_answer(q)
            print("\nAnswer:\n", ans)
            print("\nSources:")
            for u in used:
                print(f" • {u['name']}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
