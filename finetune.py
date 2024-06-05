
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM



# Load dataset
df = pd.read_excel("GSM8k_8_2.xlsx")

df.drop(["Unnamed: 0"] , axis =1 , inplace = True)

df.head()


train_dataset = Dataset.from_pandas(df[:1000])


# Formatting function for LLaMA-3

def formatting_prompts_func_llama3(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f'''<S> <|system|> You are a helpful AI assistant for solving mathematical logical reasoning tasks. You break down the problem into smaller logical steps and provide a detailed explanation to arrive at the final answer.\n<|endoftext|>\n<|user|>\n{example["question"][i]}\n<|endoftext|>\n<|assistant|> #Answer: {example["answer"][i]}'''
        output_texts.append(text)
    return output_texts

response_template = "#Answer:"

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Fine-tuned model name
new_model = "Llama-3-8B-IRIT-GSM"

# QLoRA parameters
# LoRA attention dimension
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 50

# SFT parameters
max_seq_length = None
packing = False
device_map = {"": 0}

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True ,truncation = True , add_eos_token = True , add_special_tokens = True)
tokenizer.pad_token = '#_***'
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

collator = DataCollatorForCompletionOnlyLM(response_template , tokenizer)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func_llama3,
    data_collator=collator,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

trainer.model.save_pretrained(new_model)

del model
del trainer
import gc
gc.collect()
gc.collect()

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True ,truncation = True , add_eos_token = True , add_special_tokens = True)
tokenizer.pad_token = '#_***'
tokenizer.padding_side = "right"


# Push model to Hugging Face Hub
os.system('huggingface-cli login --token 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')

tokenizer.push_to_hub("Krish2002/Llama-7B-IRIT-GSM", check_pr=True)
model.push_to_hub("Krish2002/Llama-7B-IRIT-GSM", check_pr=True)

