"""
MessagesPlaceholder was primarily designed for: flexibly handling conversation context of any length
including for few-shot prompting.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
  ChatPromptTemplate,
  MessagesPlaceholder
)
from langchain_core.messages import HumanMessage

MODEL = "qwen3:latest"

model = ChatOllama(
  model = MODEL,
  base_url = "http://localhost:11434",
)

systemPrompt = """
  You are an expert in explaining Einstein's relativity to
  non-specialist audiences.
  """

prompt_template = ChatPromptTemplate([
  ("system", systemPrompt),
  MessagesPlaceholder("message")
])

prompt = prompt_template.invoke(
  {
    "message": [
      HumanMessage("How would you explain time dilation?")
    ]
  }
)

response = model.invoke(prompt).content

print(response)