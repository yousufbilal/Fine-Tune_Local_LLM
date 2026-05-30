import torch
import transformers
import datasets
import trl
import json

file = json.load(open("caveman_dataset.json", "r"))


def format_prompt(example):
    return f"### Instruction: {example['instruction']}\n### Response: {example['output']}"

formatted_data = []

for item in file:
    formatted_item = format_prompt(item)
    formatted_data.append(formatted_item)

    # formatted_item = format_prompt(item)
    # formatted_data.append(formatted_item)

# print(formatted_data[0])
print(formatted_item)
