import torch
import transformers
import datasets
import trl
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import pipeline


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

dataset = Dataset.from_dict({"text": formatted_data})

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        num_train_epochs=15,
        per_device_train_batch_size=2,
        logging_steps=10,
        report_to="none",
        dataset_text_field="text",
        max_length=512,
    ),
)

trainer.train()

trainer.save_model("outputs/final_model")
tokenizer.save_pretrained("outputs/final_model")

pipe = pipeline("text-generation", model="outputs/final_model", tokenizer=tokenizer)

result = pipe("### Instruction: How do you make fire?\n### Response:", max_new_tokens=100)
print(result[0]["generated_text"])

