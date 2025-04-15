from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model and tokenizer from the saved directory.
model_path = "./model_finetuned"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define a prompt to start the conversation.
prompt = "User: Hi Sylvie, what can you do?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate a reply using the model.
# You can adjust the parameters (like max_new_tokens) as desired.
outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model reply:", reply)
