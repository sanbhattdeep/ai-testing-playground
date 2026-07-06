from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

env = load_dotenv(".env")

MODEL_001 = "qwen2.5:latest"
MODEL_002 = "llama3:latest"

model_001 = ChatOllama(
  base_url="http://localhost:11434",
  model=MODEL_001    
)

model_002 = ChatOllama(
  base_url="http://localhost:11434",
  model=MODEL_002
)

prompt_template = ChatPromptTemplate([
  ("system", "You are an expert in entertainment history."),
  ("human", "Provide a list of {concept} along with budget details.")
])

response_chain = prompt_template | model_001 | StrOutputParser()

title_chain = ChatPromptTemplate.from_template("Get me just the titles from the {response}.")

full_chain = {"response": response_chain} | title_chain | model_002 | StrOutputParser()

parallel_run = RunnableParallel(chain1=response_chain, chain2=full_chain)

response = parallel_run.invoke({"concept": "box office disasters"})

print(response["chain1"])
print(response["chain2"])