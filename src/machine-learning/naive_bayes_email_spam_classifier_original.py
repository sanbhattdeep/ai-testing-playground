import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

emails = [
    "Greetings! This is a legitimate email.",
    "Get rich quick! Guaranteed FREE money!",
    "We will never ask for your password.",
    "Average cashout time is 15 min!!!",
]

labels = np.array([0, 1, 0, 1])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.5, stratify=labels, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Accuracy Measure:", accuracy)
print("Precision Measure:", precision)
print("Recall Measure:", recall)
print("F1 Score:", f1)