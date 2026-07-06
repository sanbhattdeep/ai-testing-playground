from dotenv import load_dotenv
from langchain_ollama import ChatOllama

env = load_dotenv(".env")

MODEL = "qwen3:latest"

model = ChatOllama(
    model=MODEL,
    base_url="http://localhost:11434",
    temperature=0.2,
    top_k=25,
)

messages = [    
    ("system", "Provide just the values to the human questions."),
    ("human", "What is the speed of light in a vacuum?"),
    ("ai", "299,792,458 m/s"),
    ("human", "What is the speed of light in water?"),
    ("ai", "225,000,000 m/s"),
    ("human", "What is the ratio between those values?")
]

response = model.invoke(messages).content

print(response)