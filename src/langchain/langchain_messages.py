from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

MODEL = "qwen2.5:latest"

model = ChatOllama(
  model=MODEL,
  base_url="http://localhost:11434",
)

print("Testing Ollama API call (LangChain messages, no chaining)")
print("=" * 60)

messages = [
  HumanMessage(content="How many of planet Earth could fit inside Jupiter?")
]

response = model.invoke(messages)

print(response)