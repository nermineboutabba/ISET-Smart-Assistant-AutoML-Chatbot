from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset
import torch
import os
from accelerate import Accelerator

# Check PyTorch and CUDA availability
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

# Set the device to GPU 1 (second GPU, zero-indexed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Ensure GPU 1 is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("json", data_files="finetune_dataset.jsonl", split="train")

# Load tokenizer and model
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# Freeze all layers except the last one
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer
for param in model.lm_head.parameters():
    param.requires_grad = True

# Initialize accelerator to manage resources across devices
accelerator = Accelerator()

# Tokenization function with reduced max_length
def tokenize_function(example):
    prompt = f"### Question: {example['prompt']}\n### Answer: {example['completion']}"
    tokenized = tokenizer(prompt, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
    tokenized["labels"] = tokenized["input_ids"].clone()  # Copy input_ids to labels
    return tokenized

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, remove_columns=["prompt", "completion"], batched=True)

# Training arguments with reduced batch size and increased gradient accumulation
training_args = TrainingArguments(
    output_dir="./falconfinetuned",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduced batch size
    gradient_accumulation_steps=4,  # Accumulate gradients over more steps
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # Use FP16 if supported by GPU
    report_to="none",  # Disable wandb or other reporting
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# Ensure the model is moved to the correct device
model = model.to(device)

# Start training with the Accelerator managing memory and devices
trainer.train()
