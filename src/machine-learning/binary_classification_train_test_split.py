import numpy as np

# Used to divide our dataset into training data and testing data
from sklearn.model_selection import train_test_split

# LogisticRegression is a machine-learning classification algorithm
from sklearn.linear_model import LogisticRegression

# Metrics used to evaluate how well our model performs
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# 1. CREATE SOME SAMPLE DATA
# ---------------------------------------------------------

# Create 100 examples, each containing 2 input features.
#
# np.random.rand(100, 2) creates a matrix with:
#   100 rows     -> 100 observations/examples
#   2 columns    -> 2 features for each example
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


# Create the labels (the correct answers).
#
# np.random.randint(0, 2, 100) creates 100 random values
# where each value is either 0 or 1.
#
# For example:
#
# y = [0, 1, 1, 0, 0, 1, ...]
#
# This makes this a BINARY CLASSIFICATION problem:
# class 0 vs class 1.
y = np.random.randint(0, 2, 100)


# ---------------------------------------------------------
# 2. SPLIT THE DATA INTO TRAINING AND TESTING SETS
# ---------------------------------------------------------

# We should not evaluate a model using the same data
# that it was trained on.
#
# train_test_split divides the dataset into:
#
# X_train -> input features used to train the model
# X_test  -> input features used to test the model
# y_train -> correct labels used during training
# y_test  -> correct labels used during testing
#
# test_size=0.2 means:
#   80% of the data is used for training
#   20% of the data is used for testing
#
# Since we have 100 examples:
#   Training set = 80 examples
#   Test set     = 20 examples
#
# random_state=42 makes the split reproducible.
# Running the program again will create the same train/test split
# for the same X and y.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------------
# 3. CREATE THE MACHINE-LEARNING MODEL
# ---------------------------------------------------------

# Create a Logistic Regression classifier.
#
# Despite the name "Regression", Logistic Regression is commonly
# used for classification problems such as:
#
#   spam / not spam
#   fraud / not fraud
#   pass / fail
#   disease / no disease
#
model = LogisticRegression()


# ---------------------------------------------------------
# 4. TRAIN THE MODEL
# ---------------------------------------------------------

# fit() trains the model.
#
# The model receives:
#
# X_train -> input features
# y_train -> correct answers
#
# It tries to learn a relationship such as:
#
#     features -> class
#
# In a real dataset, it might learn:
#
# age + income -> likely to buy / not buy
#
# In this example, however, both X and y are random,
# so there isn't actually a meaningful relationship to learn.
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 5. MAKE PREDICTIONS ON UNSEEN TEST DATA
# ---------------------------------------------------------

# predict() asks the trained model to predict the class
# for each example in X_test.
#
# Since X_test contains 20 examples, y_pred will contain
# 20 predictions.
#
# Example:
#
# y_pred = [0, 1, 0, 0, 1, ...]
#
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 6. EVALUATE THE MODEL
# ---------------------------------------------------------

# Accuracy measures:
#
#     number of correct predictions
#     -----------------------------
#     total number of predictions
#
# Example:
# If 15 out of 20 predictions are correct:
#
# accuracy = 15 / 20 = 0.75
#
# or 75%.
accuracy = accuracy_score(y_test, y_pred)


# F1 score combines two other metrics:
#
# Precision -> Of everything predicted as positive,
#              how many were actually positive?
#
# Recall    -> Of all actual positives,
#              how many did the model correctly find?
#
# F1 combines precision and recall into one score.
#
# F1 ranges roughly from:
#
# 0 -> poor
# 1 -> perfect
#
# F1 is particularly useful when the classes are imbalanced
# or when both false positives and false negatives matter.
f1 = f1_score(y_test, y_pred)


# ---------------------------------------------------------
# 7. PRINT THE RESULTS
# ---------------------------------------------------------

print("Accuracy Measure:", accuracy)
print("F1 Score:", f1)