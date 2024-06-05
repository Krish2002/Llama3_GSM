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


# Load dataset
df = pd.read_excel("GSM8k_8_2.xlsx")
df.drop(["Unnamed: 0"] , axis =1 , inplace = True)
df.head()
test_dataset = Dataset.from_pandas(df[1000:])

# Inferece

access_token = ""
model_name = "Llama-3-8B-IRIT-GSM"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)

def generate_response(setup_prompt):
  model_inputs = tokenizer(setup_prompt,return_tensors = "pt").to("cuda:0")
  output = model.generate(**model_inputs , max_new_tokens = 400 , eos_token_id= tokenizer.eos_token_id)
  question_to_claims = tokenizer.decode(output[0], skip_special_tokens=True)
  prompt_tokens = len(setup_prompt.split())
  response = ' '.join(question_to_claims.split()[prompt_tokens:])
  return response

def naive_prompt(question):
  prompt = f'''<S> <|system|> You are a helpful AI assistant for solving mathematical logical reasoning tasks. You break down the problem into smaller logical steps and provide a detailed explanation to arrive at the final answer.\n<|endoftext|>\n<|user|>\n{question}\n<|endoftext|>\n<|assistant|> #Answer:'''
  return prompt