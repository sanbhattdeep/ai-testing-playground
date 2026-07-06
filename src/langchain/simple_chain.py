from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = "qwen3:latest"

model = ChatOllama(
  model=MODEL,
  base_url="http://localhost:11434",
)

system_prompt = """
  You are an expert in explaining Einstein's relativity to
  non-specialist audiences.
  """

prompt_template = ChatPromptTemplate([
  ("system", system_prompt),
  ("human", "How would you explain {concept}?")
])

chain = prompt_template | model | StrOutputParser()
response = chain.invoke({"concept": "time dilation"})

print(response)