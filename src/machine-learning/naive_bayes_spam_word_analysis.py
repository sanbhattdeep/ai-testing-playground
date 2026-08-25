import numpy as np

# CountVectorizer converts text documents into numerical
# word-count features that a machine-learning model can use.
from sklearn.feature_extraction.text import CountVectorizer

# Multinomial Naive Bayes is commonly used for classification
# tasks involving counts, especially text classification.
from sklearn.naive_bayes import MultinomialNB

# Metrics used to evaluate the classifier.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Used to split the dataset into training and test sets.
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. CREATE A SMALL EMAIL DATASET
# ---------------------------------------------------------

# Each string represents one email.
#
# In this example:
#
#   Class 0 -> legitimate / non-spam email ("ham")
#   Class 1 -> spam email
emails = [
    "Greetings! This is a legitimate email.",
    "Get rich quick! Guaranteed FREE money!",
    "We will never ask for your password.",
    "Average cashout time is 15 min!!!",
]


# ---------------------------------------------------------
# 2. CREATE THE LABELS
# ---------------------------------------------------------

# Each label corresponds to the email at the same position.
#
#   0 -> legitimate / non-spam
#   1 -> spam
#
# Therefore:
#
# Email 1 -> 0
# Email 2 -> 1
# Email 3 -> 0
# Email 4 -> 1
labels = np.array([0, 1, 0, 1])


# ---------------------------------------------------------
# 3. CONVERT THE EMAIL TEXT INTO NUMERICAL FEATURES
# ---------------------------------------------------------

# Create a CountVectorizer.
#
# The vectorizer builds a vocabulary containing the words
# found in the emails and represents each email using
# word counts.
#
# Example idea:
#
# Vocabulary:
#
#     free   money   password   email
#
# Email:
#
#     "free money free"
#
# could become:
#
#     [2, 1, 0, 0]
vectorizer = CountVectorizer()


# fit_transform() does two things:
#
# 1. fit()
#    Learns the vocabulary from all of the emails.
#
# 2. transform()
#    Converts each email into numerical word-count features.
#
# X is therefore no longer raw text.
# It is a sparse numerical matrix.
#
# IMPROVEMENT NOTE:
# In a real evaluation, it is generally better to split the
# raw emails into training/test sets BEFORE fitting the vectorizer.
#
# Otherwise the vectorizer learns the vocabulary from the test
# emails too, which is a small form of data leakage.
#
# The original script is intentionally left unchanged here.
X = vectorizer.fit_transform(emails)


# ---------------------------------------------------------
# 4. SPLIT DATA INTO TRAINING AND TEST SETS
# ---------------------------------------------------------

# Split the vectorized emails and labels into training
# and test data.
#
# test_size=0.5 means:
#
#   50% training
#   50% testing
#
# Since there are only 4 emails:
#
#   2 emails -> training
#   2 emails -> testing
#
# stratify=labels tries to preserve the class proportions.
#
# Since the full dataset contains:
#
#   2 legitimate emails
#   2 spam emails
#
# we want approximately:
#
# Training:
#   1 legitimate
#   1 spam
#
# Testing:
#   1 legitimate
#   1 spam
#
# random_state=42 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.5, stratify=labels, random_state=42
)


# ---------------------------------------------------------
# 5. CREATE THE CLASSIFICATION MODEL
# ---------------------------------------------------------

# Create a Multinomial Naive Bayes classifier.
#
# This algorithm works particularly well with features
# representing counts, such as word counts.
model = MultinomialNB()


# ---------------------------------------------------------
# 6. TRAIN THE MODEL
# ---------------------------------------------------------

# Train the classifier using:
#
# X_train -> word-count features
# y_train -> correct spam/non-spam labels
#
# During training, the model learns things such as:
#
# P(word | spam)
#
# and:
#
# P(word | non-spam)
#
# which means:
#
# "How likely is this word to occur in emails belonging
# to a particular class?"
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 7. MAKE PREDICTIONS
# ---------------------------------------------------------

# Predict whether each test email belongs to:
#
#   0 -> non-spam
#   1 -> spam
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 8. CALCULATE MODEL EVALUATION METRICS
# ---------------------------------------------------------

# Accuracy:
#
# Of ALL test emails, what fraction did the model
# classify correctly?
accuracy = accuracy_score(y_test, y_pred)


# Precision:
#
# Of all emails predicted as SPAM,
# what fraction actually were spam?
#
# Formula:
#
#               TP
# Precision = --------
#             TP + FP
#
# zero_division=0 returns 0 if the model predicts
# no positive examples.
precision = precision_score(y_test, y_pred, zero_division=0)


# Recall:
#
# Of all emails that ACTUALLY WERE spam,
# what fraction did the model successfully detect?
#
# Formula:
#
#            TP
# Recall = --------
#          TP + FN
recall = recall_score(y_test, y_pred, zero_division=0)


# F1 score:
#
# Combines precision and recall into one metric.
#
# Formula:
#
#              Precision * Recall
# F1 = 2 * -------------------------
#              Precision + Recall
f1 = f1_score(y_test, y_pred, zero_division=0)


# Print the evaluation metrics.
print("Accuracy Measure:", accuracy)
print("Precision Measure:", precision)
print("Recall Measure:", recall)
print("F1 Score:", f1)


# ---------------------------------------------------------
# 9. GET THE WORDS LEARNED BY COUNTVECTORIZER
# ---------------------------------------------------------

# get_feature_names_out() returns the vocabulary created
# by CountVectorizer.
#
# For example:
#
# [
#     "average",
#     "cashout",
#     "email",
#     "free",
#     "money",
#     ...
# ]
#
# Each word corresponds to one column in X.
feature_names = vectorizer.get_feature_names_out()


# ---------------------------------------------------------
# 10. GET WORD PROBABILITIES LEARNED FOR THE SPAM CLASS
# ---------------------------------------------------------

# MultinomialNB stores its learned word probabilities in:
#
#     model.feature_log_prob_
#
# Shape conceptually:
#
#                 word1   word2   word3   ...
#
# Class 0       [ ...     ...     ...   ]
# Class 1       [ ...     ...     ...   ]
#
#
# [1] selects the learned values for:
#
#     Class 1 -> spam
#
#
# IMPORTANT:
# These values are LOG probabilities, not ordinary
# probabilities.
#
# Conceptually they represent:
#
#     log(P(word | spam))
#
spam_word_probabilities = model.feature_log_prob_[1]


# ---------------------------------------------------------
# 11. GET WORD PROBABILITIES FOR NON-SPAM / HAM
# ---------------------------------------------------------

# [0] selects Class 0.
#
# These values represent:
#
#     log(P(word | non-spam))
#
# "Ham" is commonly used in spam-filtering terminology
# to mean legitimate / non-spam email.
ham_word_probabilities = model.feature_log_prob_[0]


# ---------------------------------------------------------
# 12. FIND THE TOP FIVE WORDS IN THE SPAM CLASS
# ---------------------------------------------------------

print("\nTop five words indicating spam:")


# np.argsort() returns the indexes that would sort
# the array from smallest to largest.
#
# Example:
#
# values:
#
#     [0.3, 0.8, 0.2]
#
# np.argsort(values):
#
#     [2, 0, 1]
#
# because:
#
# values[2] = 0.2
# values[0] = 0.3
# values[1] = 0.8
#
#
# [-5:] selects the indexes of the five largest values.
#
# [::-1] reverses their order so that the largest value
# comes first.
#
# The result therefore contains the indexes of the five
# words having the highest P(word | spam).
top_spam_words = np.argsort(spam_word_probabilities)[-5:][::-1]


# Loop through those word indexes.
for idx in top_spam_words:

    # feature_names[idx] gives us the actual word.
    #
    # spam_word_probabilities[idx] is stored as a
    # LOG probability.
    #
    # np.exp() converts the log probability back into
    # a normal probability.
    #
    # If:
    #
    #     log(P) = -1.5
    #
    # then:
    #
    #     P = exp(-1.5)
    #
    print(f"{feature_names[idx]} (Probability: {np.exp(spam_word_probabilities[idx])})")


# ---------------------------------------------------------
# 13. FIND THE TOP FIVE WORDS IN THE NON-SPAM CLASS
# ---------------------------------------------------------

print("\nTop five words indicating non-spam:")


# Same process as above, but now we examine:
#
#     P(word | non-spam)
#
# rather than:
#
#     P(word | spam)
top_ham_words = np.argsort(ham_word_probabilities)[-5:][::-1]


# Print each highly probable non-spam word.
for idx in top_ham_words:
    print(f"{feature_names[idx]} (Probability: {np.exp(ham_word_probabilities[idx])})")