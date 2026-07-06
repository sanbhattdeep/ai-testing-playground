from langchain_ollama import ChatOllama
from langchain_core.prompts import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate
)

model = ChatOllama(
  base_url="http://localhost:11434",
  model="qwen2.5:latest"
)

systemPrompt = "You are an expert in explaining Einstein's relativity to non-specialist audiences."
humanPrompt = "How would you explain {concept}?"

systemMessage = SystemMessagePromptTemplate.from_template(systemPrompt)
humanMessage = HumanMessagePromptTemplate.from_template(humanPrompt)

prompt_template = ChatPromptTemplate([
  systemMessage,
  humanMessage
])

prompt = prompt_template.invoke({"concept": "time dilation"})

response = model.invoke(prompt).content

print(response)