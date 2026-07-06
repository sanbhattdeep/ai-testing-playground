from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

MODEL = "qwen2.5:latest"

model = ChatOllama(
  model = MODEL,
  base_url = "http://localhost:11434",
)

raw_prompt = "How many of planet Earth could fit inside {planet}?"

prompt_template = ChatPromptTemplate.from_template(raw_prompt)

full_prompt = prompt_template.format(planet="Jupiter")

response = model.invoke(full_prompt).content

print(response)