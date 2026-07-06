from dotenv import load_dotenv
from langchain_ollama import ChatOllama

env = load_dotenv(".env")

MODEL = "qwen3:latest"

model = ChatOllama(
    model=MODEL,
    base_url="http://localhost:11434",
    temperature=0.2,
    top_k=25
)

prompt = "What is the speed of light in a vacuum?"

response = model.invoke(prompt)

print(type(response))
print(response)