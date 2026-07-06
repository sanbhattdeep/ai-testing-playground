from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = "qwen3:latest"

model = ChatOllama(
  model=MODEL,
  base_url="http://localhost:11434",
)

prompt_template = ChatPromptTemplate([
  ("system", "You are an expert in entertainment history."),
  ("human", "Provide a list of {concept} along with budget details.")
])

response_chain = prompt_template | model | StrOutputParser()
 
title_chain = ChatPromptTemplate.from_template("Get me just the titles from the {response}.")
 
full_chain = {"response": response_chain} | title_chain | model | StrOutputParser()
 
response = full_chain.invoke({"concept": "box office disasters"})

print(response)