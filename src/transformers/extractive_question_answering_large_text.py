import pandas as pd
from transformers import pipeline

with open("course-text.txt", "r", encoding="utf-8") as file:
    text = file.read() 

reader = pipeline("question-answering")

question = "What did King Hezekiah do?"

outputs = reader(question=question, context=text)

df = pd.DataFrame([outputs])
print(df)