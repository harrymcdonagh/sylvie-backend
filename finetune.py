from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType
)

# 1) Load & format dataset
dataset = load_dataset("json", data_files="newdata.jsonl")

def format_prompt(example):
    dialogue = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        dialogue += f"<|{role}|>\n{content}\n"
    return {"text": dialogue}

dataset = dataset.map(format_prompt, remove_columns=["messages"])

# 2) Tokenizer & long-context tokenization
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,      # ← increased from 512
    )

tokenized = dataset.map(tokenize_fn, batched=True)

# 3) Load quantized model & prepare for LoRA
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)
# prepare internal modules for k-bit training
base_model = prepare_model_for_kbit_training(base_model)

# 4) Attach LoRA adapters
lora_config = LoraConfig(
    r=32,                     # higher capacity
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)

# 5) Training arguments with gradient accumulation
training_args = TrainingArguments(
    output_dir="./sylvie-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch size = 16
    num_train_epochs=5,
    learning_rate=1e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# 6) Data collator (causal LM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 7) Fine-tune & save only the LoRA adapters
trainer.train()
model.save_pretrained("./sylvie-lora")
