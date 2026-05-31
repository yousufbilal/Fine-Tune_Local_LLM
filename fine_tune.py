import torch
import transformers
import datasets
import trl
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

#im am fine tuniing a instruct model not a base model, I just need to make sure the data is in the right format for the instruct model
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

file = json.load(open("caveman_dataset.json", "r"))


def format_prompt(example):
    return f"### Instruction: {example['instruction']}\n### Response: {example['output']}"

formatted_data = []

for item in file:
    formatted_item = format_prompt(item)
    formatted_data.append(formatted_item)



tokenizer = AutoTokenizer.from_pretrained(model_name)

# downloading the model
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded!")



