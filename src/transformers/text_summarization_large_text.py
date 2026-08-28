from transformers import pipeline

with open("course-text.txt", "r", encoding="utf-8") as file:
    text = file.read()

summarizer = pipeline("summarization")
trimmed_text = text[:1024]
outputs = summarizer(trimmed_text, max_length=45, clean_up_tokenization_spaces=True)

print(outputs[0]["summary_text"])