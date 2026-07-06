from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

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
  ("placeholder", "{conversation_history}"),
  ("human", "{current_question}")
])

prompt1 = prompt_template.invoke({
  "conversation_history": [],
  "current_question": "How would you explain time dilation?"
})

response1 = model.invoke(prompt1).content

print("FIRST RESPONSE:")
print(response1)

prompt2 = prompt_template.invoke({
  "conversation_history": [
    HumanMessage("How would you explain time dilation?"),
    AIMessage(response1)
  ],
  "current_question": "Can you give me a concrete example involving twins?"
})

response2 = model.invoke(prompt2).content

print("\nSECOND RESPONSE:")
print(response2)