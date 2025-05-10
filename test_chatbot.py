from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# === Load model and tokenizer ===
BASE_MODEL_PATH = "./model"
LORA_ADAPTER_PATH = "./sylvie-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map="auto")
model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
model.eval()

# === Generate response ===
def chat(
    user_input,
    system_message="You are a helpful student support assistant at UEA. Answer students clearly and kindly."
):
    prompt = (
        f"<|system|>\n{system_message}\n"
        f"<|user|>\n{user_input}\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.75,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the assistant's reply
    if "<|assistant|>" in decoded:
        assistant_part = decoded.split("<|assistant|>")[-1]
        # Stop at next tag if it exists
        for stop_tag in ["<|user|>", "<|system|>"]:
            if stop_tag in assistant_part:
                assistant_part = assistant_part.split(stop_tag)[0]
        return assistant_part.strip()

    return decoded.strip()

# === Interactive CLI ===
if __name__ == "__main__":
    print("Hi! I'm Sylvie, your UEA student support assistant. Ask me anything or type 'exit' to quit.\n")
    while True:
        user_input = input("Student: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Chatbot: Take care! 👋")
            break
        response = chat(user_input)
        print(f"Chatbot: {response}\n")
