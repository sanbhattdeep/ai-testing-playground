import pytest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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

@pytest.mark.parametrize(
    "email, expected_label",
    [
        ("Hello, checking in to see how you are.", 0),
        ("URGENT: Claim your free prize now!", 1),
        ("Important account information update", 0),
        ("Exclusive limited-time offer!", 1),
    ],
)
def test_spam_classification(email, expected_label):
    email_vector = vectorizer.transform([email])

    predicted_label = model.predict(email_vector)

    assert predicted_label == expected_label