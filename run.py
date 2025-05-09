import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from kb_query import get_relevant_services
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
    # 1) KB lookup (only services with cosine ≥ threshold)
    hits = get_relevant_services(question, top_k=top_k, min_score=0.6)

    # 2) Build prompt; include services block only if we have hits
    if hits:
        services_block = "\n".join(
            f"- {h['name']}: {h['desc']} ({h['url']})"
            for h in hits
        )
        prompt_context = f"Here are some UEA services that might help:\n{services_block}\n\n"
    else:
        prompt_context = (
            "I don’t have a specific UEA service to recommend for that, "
            "but I can suggest some strategies you might try:\n\n"
        )

    prompt = f"""
System: You are Sylvie, a compassionate mental-health support chatbot for UEA students. Provide direct, complete answers. Do NOT ask follow-up or rhetorical questions. Your goals are to:  
  1. Greet the student warmly and by name (if known).  
  2. Reflect back the emotion you sense (“I’m really sorry to hear you’re feeling X…”).  
  3. Validate it (“What you’re going through makes total sense…”).  
  4. Offer 1–2 concrete coping strategies or self-care tips.  
  5. Casually recommend 1–3 relevant UEA services drawn naturally from the context below.  
  6. If the student mentions crisis, self-harm, or suicidal thoughts, immediately include:  
     “If you ever feel like you might act on these thoughts, please reach out right now to Nightline (https://norwich.nightline.ac.uk/) or call 999/111 — you’re not alone and help is here.”  
  7. Invite them to keep asking questions or check back anytime.  

{prompt_context}
User: {question}  
Assistant:""".strip()

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
        num_beams=1,
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
            if used:
                print("\nSources:")
                for u in used:
                    print(f" • {u['name']}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
