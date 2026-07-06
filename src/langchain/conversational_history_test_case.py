from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

env = load_dotenv(".env")
# dictionary we’ll use to hold conversation state in memory
store = {}
session_id = "jeff-chat"

MODEL = "qwen3:latest"
USE_SQLITE = False
DB = "jeff-chat.db"

# ============================================================
# SESSION HISTORY MANAGEMENT
# ============================================================
 
def read_session_history(session_id: str) -> BaseChatMessageHistory:
  if USE_SQLITE:
    return SQLChatMessageHistory(
      session_id=session_id,
      connection=f"sqlite:///{DB}"
    )
  else:
    if (session_id not in store):
      store[session_id] = ChatMessageHistory()
 
    return store[session_id]

read_session_history(session_id).clear()

# ============================================================
# MODEL SETUP
# ============================================================

model = ChatOllama(
  model=MODEL,
  base_url="http://localhost:11434",
)

template = ChatPromptTemplate.from_messages([
  ("system", "Please answer as concisely as possible."),
  ("placeholder", "{history}"),
  ("human", "{prompt}")
])

chain = template | model | StrOutputParser()

history = RunnableWithMessageHistory(
  chain,
  read_session_history,
  input_messages_key="prompt",
  history_messages_key="history"
)

# ============================================================
# CONVERSATION EXECUTION
# ============================================================
 
response1 = history.invoke(
  {"prompt": "What is the smallest possible length?"},
  config={"configurable": {"session_id": session_id}}
)
 
response2 = history.invoke(
  {"prompt": "What is the smallest possible time?"},
  config={"configurable": {"session_id": session_id}}
)
 
response3 = history.invoke(
  {"prompt": "Do those values define the minimal scale of physical events?"},
  config={"configurable": {"session_id": session_id}}
)
 
print(response1, end="\n\n")
print(response2, end="\n\n")
print(response3)