from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

MODEL = "qwen3:latest"

model = ChatOllama(
  model = MODEL,
  base_url = "http://localhost:11434",
)

prompt_template = ChatPromptTemplate([
  ("system", "You are an expert in explaining Einstein's relativity to non-specialist audiences."),
  ("human", "How would you explain {concept}?")
])

prompt = prompt_template.invoke({"concept": "time dilation"})

response = model.invoke(prompt).content

print(response)