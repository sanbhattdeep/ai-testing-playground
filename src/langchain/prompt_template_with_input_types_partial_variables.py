from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

MODEL = "qwen2.5:latest"

model = ChatOllama(
  model = MODEL,
  base_url = "http://localhost:11434",
)

raw_prompt = "What is the mass of the {particle}? Answer in {units}."

prompt_template = ChatPromptTemplate.from_template(
  raw_prompt,
  input_types={"particle": str},
  partial_variables={"units": "kilograms"}
)

#full_prompt = prompt_template.format(particle="electron")
#make it Runnable
full_prompt = prompt_template.invoke({"particle": "electron"})

response = model.invoke(full_prompt).content

print(full_prompt)
print(response)