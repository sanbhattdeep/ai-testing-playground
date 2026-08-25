import numpy as np

# CountVectorizer converts text into numerical features
# by counting how many times each word appears.
from sklearn.feature_extraction.text import CountVectorizer

# Multinomial Naive Bayes is a classification algorithm
# commonly used for text classification problems such as:
#
#   spam / not spam
#   positive / negative sentiment
#   topic classification
from sklearn.naive_bayes import MultinomialNB

# Metrics used to evaluate the classifier.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Used to split the dataset into training and test sets.
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. CREATE A SMALL EMAIL DATASET
# ---------------------------------------------------------

# Each string represents one email.
#
# In this example:
#
#   legitimate emails -> Class 0
#   spam emails       -> Class 1
#
emails = [
    "Greetings! This is a legitimate email.",
    "Get rich quick! Guaranteed FREE money!",
    "We will never ask for your password.",
    "Average cashout time is 15 min!!!",
]


# ---------------------------------------------------------
# 2. CREATE THE LABELS
# ---------------------------------------------------------

# Each number corresponds to the email at the same position.
#
#   0 -> legitimate email
#   1 -> spam email
#
# Therefore:
#
# Email 1 -> 0 -> legitimate
# Email 2 -> 1 -> spam
# Email 3 -> 0 -> legitimate
# Email 4 -> 1 -> spam
#
labels = np.array([0, 1, 0, 1])


# ---------------------------------------------------------
# 3. SPLIT THE RAW EMAILS INTO TRAINING AND TEST SETS
# ---------------------------------------------------------

# Unlike our previous numeric examples, we split the RAW TEXT
# before converting it into numerical features.
#
# This avoids data leakage.
#
# test_size=0.5 means:
#
#   50% training data
#   50% test data
#
# Since there are only 4 emails:
#
#   2 emails -> training
#   2 emails -> testing
#
#
# stratify=labels is particularly important here.
#
# It tries to preserve the proportion of classes in
# both the training and test sets.
#
# Since we have:
#
#   2 legitimate emails
#   2 spam emails
#
# the split will contain approximately:
#
# Training:
#   1 legitimate
#   1 spam
#
# Test:
#   1 legitimate
#   1 spam
#
#
# random_state=42 makes the split reproducible.
emails_train, emails_test, y_train, y_test = train_test_split(
    emails,
    labels,
    test_size=0.5,
    stratify=labels,
    random_state=42,
)


# ---------------------------------------------------------
# 4. CREATE THE TEXT VECTORIZER
# ---------------------------------------------------------

# Machine-learning models cannot directly understand strings
# such as:
#
#     "Get rich quick! Guaranteed FREE money!"
#
# CountVectorizer converts the text into numbers based on
# word occurrence counts.
#
# For example, it may build a vocabulary containing words like:
#
#     cashout
#     email
#     free
#     guaranteed
#     money
#     password
#     quick
#     rich
#
vectorizer = CountVectorizer()


# ---------------------------------------------------------
# 5. LEARN THE VOCABULARY FROM TRAINING EMAILS
# ---------------------------------------------------------

# fit_transform() performs TWO operations:
#
# 1. fit()
#    Learn the vocabulary from the TRAINING emails.
#
# 2. transform()
#    Convert those training emails into numerical vectors.
#
# IMPORTANT:
# We intentionally fit only on the training data.
#
# The vectorizer must NOT learn anything from the test data
# before evaluation.
X_train = vectorizer.fit_transform(emails_train)


# ---------------------------------------------------------
# 6. TRANSFORM THE TEST EMAILS
# ---------------------------------------------------------

# transform() converts the test emails into numerical vectors
# using ONLY the vocabulary already learned from the
# training emails.
#
# Notice that we do NOT call fit_transform() here.
#
# The test data should remain unseen while the system
# is learning.
X_test = vectorizer.transform(emails_test)


# ---------------------------------------------------------
# 7. CREATE THE CLASSIFICATION MODEL
# ---------------------------------------------------------

# Create a Multinomial Naive Bayes classifier.
#
# MultinomialNB works particularly well with count-based
# features such as those produced by CountVectorizer.
model = MultinomialNB()


# ---------------------------------------------------------
# 8. TRAIN THE MODEL
# ---------------------------------------------------------

# Train the classifier using:
#
# X_train -> numerical word-count features
# y_train -> correct spam/legitimate labels
#
# The model attempts to learn patterns such as:
#
# "free", "money", "cashout"
#
# potentially being associated with spam.
#
# With only 2 training emails, however, this model has
# almost no meaningful data to learn from.
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 9. MAKE PREDICTIONS
# ---------------------------------------------------------

# Predict whether each unseen test email is:
#
#   0 -> legitimate
#   1 -> spam
#
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 10. CALCULATE ACCURACY
# ---------------------------------------------------------

# Accuracy answers:
#
# "Of ALL the test emails, what proportion did the
# model classify correctly?"
#
# Formula:
#
#               Correct predictions
# Accuracy = -----------------------------
#              Total predictions
#
accuracy = accuracy_score(y_test, y_pred)


# ---------------------------------------------------------
# 11. CALCULATE PRECISION
# ---------------------------------------------------------

# For this binary classification problem:
#
#   Class 1 = spam
#
# Precision therefore asks:
#
# "Of all the emails the model predicted as SPAM,
#  how many were actually spam?"
#
# Formula:
#
#                 True Positives
# Precision = ------------------------
#              True Positives + False Positives
#
#
# zero_division=0 prevents an error/warning if the model
# predicts no positive examples.
#
# For example:
#
# If the model predicts ZERO emails as spam,
# TP + FP would be zero.
#
# In that situation we define precision as 0.
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)


# ---------------------------------------------------------
# 12. CALCULATE RECALL
# ---------------------------------------------------------

# Recall asks:
#
# "Of all the emails that ACTUALLY WERE spam,
#  how many did the model successfully detect?"
#
# Formula:
#
#              True Positives
# Recall = ------------------------
#           True Positives + False Negatives
#
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)


# ---------------------------------------------------------
# 13. CALCULATE F1 SCORE
# ---------------------------------------------------------

# F1 combines Precision and Recall into one score.
#
# Formula:
#
#              Precision × Recall
# F1 = 2 × --------------------------
#              Precision + Recall
#
# A high F1 score means the classifier is doing reasonably
# well at BOTH:
#
#   - avoiding false spam alarms
#   - detecting actual spam
#
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)


# ---------------------------------------------------------
# 14. PRINT THE RESULTS
# ---------------------------------------------------------

print("Accuracy Measure:", accuracy)
print("Precision Measure:", precision)
print("Recall Measure:", recall)
print("F1 Score:", f1)