from transformers import pipeline

model = pipeline(
    "text-generation",
    model="outputs/final_model",
    tokenizer="outputs/final_model",
)

prompt = f"### Instruction: what is a lion?\n### Response:"
answer = model(prompt, max_new_tokens=100, repetition_penalty=1.3)
           
print(answer[0]["generated_text"]) 