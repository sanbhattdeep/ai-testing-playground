import numpy as np

# cross_val_score is used to evaluate a model using cross-validation.
# Instead of creating just one train/test split, it creates multiple
# train/test splits and evaluates the model multiple times.
from sklearn.model_selection import cross_val_score

# Logistic Regression is a classification algorithm.
# It is commonly used for binary classification problems such as:
# spam / not spam
# fraud / not fraud
# pass / fail
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# 1. CREATE SAMPLE DATA
# ---------------------------------------------------------

# Create 100 examples, each containing 2 input features.
#
# Shape of X:
#   100 rows    -> 100 observations/examples
#   2 columns   -> 2 features per example
#
# Each value is a random decimal number between 0 and 1.
#
# Example:
#
# X =
# [
#   [0.25, 0.71],
#   [0.82, 0.13],
#   [0.44, 0.91],
#   ...
# ]
#
X = np.random.rand(100, 2)


# Create 100 random labels.
#
# Each label is either:
#   0
#   1
#
# This makes this a binary classification problem.
#
# Example:
#
# y = [0, 1, 1, 0, 0, 1, ...]
#
# IMPORTANT:
# X and y are generated independently and randomly.
# Therefore, there is no meaningful relationship between the
# features and labels for the model to learn.
y = np.random.randint(0, 2, 100)


# ---------------------------------------------------------
# 2. CREATE THE MODEL
# ---------------------------------------------------------

# Create a Logistic Regression classifier.
#
# At this point the model has NOT been trained yet.
#
# cross_val_score() will train the model separately
# during each cross-validation fold.
model = LogisticRegression()


# ---------------------------------------------------------
# 3. EVALUATE ACCURACY USING 5-FOLD CROSS-VALIDATION
# ---------------------------------------------------------

# cross_val_score() performs cross-validation.
#
# cv=5 means:
#   Divide the dataset into 5 parts (called folds).
#
# With 100 examples, each fold will contain roughly 20 examples.
#
# For each round:
#
#   4 folds (~80 examples) -> training data
#   1 fold  (~20 examples) -> testing data
#
# This process is repeated 5 times so that every fold
# gets used as the test set once.
#
# scoring="accuracy" means:
# Evaluate each model using accuracy.
#
# scores will contain 5 accuracy values, one for each fold.
#
# Example:
#
# scores = [0.55, 0.45, 0.60, 0.50, 0.55]
#
scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)


# Calculate the average accuracy across all 5 folds.
#
# Example:
#
# scores = [0.55, 0.45, 0.60, 0.50, 0.55]
#
# avg_accuracy = 0.53
#
avg_accuracy = np.mean(scores)


# ---------------------------------------------------------
# 4. EVALUATE F1 SCORE USING 5-FOLD CROSS-VALIDATION
# ---------------------------------------------------------

# Run another 5-fold cross-validation.
#
# This time, instead of measuring accuracy,
# we measure the F1 score.
#
# scoring="f1" treats class 1 as the positive class.
#
# f1_scores will contain 5 F1 scores,
# one for each fold.
#
# Example:
#
# f1_scores = [0.50, 0.42, 0.57, 0.48, 0.52]
#
f1_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
)


# Calculate the average F1 score across all 5 folds.
avg_f1 = np.mean(f1_scores)


# ---------------------------------------------------------
# 5. PRINT THE FINAL RESULTS
# ---------------------------------------------------------

print("Accuracy Measure (average):", avg_accuracy)
print("F1 Score (average):", avg_f1)