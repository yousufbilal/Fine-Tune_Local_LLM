from langchain_ollama import OllamaLLM

model = OllamaLLM(model="smollm2:135m")

response = model.invoke("hello world")

print(response)