import numpy as np

# Seaborn is used here to display the confusion matrix as a heatmap
import seaborn as sns

# Matplotlib is used to label and display the heatmap
import matplotlib.pyplot as plt

# Used to divide the dataset into training and test sets
from sklearn.model_selection import train_test_split

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression

# Metrics used to evaluate the classifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ---------------------------------------------------------
# 1. CREATE SAMPLE DATA
# ---------------------------------------------------------

# Create 100 examples, each containing 2 input features.
#
# Shape of X:
#   100 rows    -> 100 observations/examples
#   2 columns   -> 2 features for each example
#
# Each value is a random decimal number between 0 and 1.
#
# Example:
#
# X =
# [
#     [0.25, 0.71],
#     [0.82, 0.13],
#     [0.44, 0.91],
#     ...
# ]
X = np.random.rand(100, 2)


# Create 100 random class labels.
#
# Each label is either:
#   0 -> Class 0
#   1 -> Class 1
#
# This makes this a binary classification problem.
#
# IMPORTANT:
# X and y are generated independently and randomly.
# Therefore, there is no meaningful relationship between
# the features and the labels for the model to learn.
y = np.random.randint(0, 2, 100)


# ---------------------------------------------------------
# 2. SPLIT DATA INTO TRAINING AND TEST SETS
# ---------------------------------------------------------

# Split the data into:
#
# X_train -> features used to train the model
# X_test  -> features used to test the model
# y_train -> true labels used during training
# y_test  -> true labels used during testing
#
# test_size=0.2 means:
#   80% of the data is used for training
#   20% is used for testing
#
# Since we have 100 examples:
#   80 examples -> training
#   20 examples -> testing
#
# random_state=42 makes the train/test split reproducible
# for the same X and y.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------------
# 3. CREATE AND TRAIN THE MODEL
# ---------------------------------------------------------

# Create a Logistic Regression classifier.
#
# At this stage, the model has not learned anything yet.
model = LogisticRegression()


# Train the model using the training data.
#
# The model looks at:
#   X_train -> input features
#   y_train -> correct answers
#
# and tries to learn how the input features relate to
# Class 0 or Class 1.
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. MAKE PREDICTIONS ON THE TEST DATA
# ---------------------------------------------------------

# Ask the trained model to predict the class of each example
# in X_test.
#
# Since X_test contains 20 examples, y_pred will contain
# 20 predictions.
#
# Example:
#
# y_pred = [0, 1, 0, 0, 1, ...]
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 5. CALCULATE ACCURACY
# ---------------------------------------------------------

# Accuracy measures the fraction of all predictions
# that were correct.
#
# Formula:
#
#                Correct predictions
# Accuracy = -----------------------------
#             Total number of predictions
#
# Example:
#
# 15 correct predictions out of 20:
#
# Accuracy = 15 / 20 = 0.75
#
# which means 75% accuracy.
accuracy = accuracy_score(y_test, y_pred)


# ---------------------------------------------------------
# 6. CALCULATE F1 SCORE
# ---------------------------------------------------------

# F1 score combines precision and recall into one metric.
#
# Precision:
#   Of everything predicted as Class 1,
#   how many really were Class 1?
#
# Recall:
#   Of all examples that really were Class 1,
#   how many did the model successfully find?
#
# F1 is useful when both false positives and
# false negatives matter.
f1 = f1_score(y_test, y_pred)


# ---------------------------------------------------------
# 7. CREATE THE CONFUSION MATRIX
# ---------------------------------------------------------

# confusion_matrix() compares:
#
#   y_test -> actual / true labels
#   y_pred -> model's predicted labels
#
# For binary classification, the result has this structure:
#
#                 Predicted
#                 0       1
#
# Actual 0       TN      FP
#
# Actual 1       FN      TP
#
# TN = True Negative
# FP = False Positive
# FN = False Negative
# TP = True Positive
#
# Example:
#
# cm =
# [
#     [8, 2],
#     [3, 7]
# ]
#
# means:
#
# 8 True Negatives
# 2 False Positives
# 3 False Negatives
# 7 True Positives
cm = confusion_matrix(y_test, y_pred)


# ---------------------------------------------------------
# 8. DISPLAY THE CONFUSION MATRIX AS A HEATMAP
# ---------------------------------------------------------

# sns.heatmap() turns the confusion matrix into a visual grid.
#
# annot=True:
#   Displays the actual number inside each square.
#
# cmap="Blues":
#   Uses different shades of blue to represent the values.
#
# fmt="d":
#   Displays the numbers as integers instead of decimals.
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)


# The columns represent what the model PREDICTED.
plt.xlabel("Predicted labels")

# The rows represent the TRUE / ACTUAL labels.
plt.ylabel("True labels")

# Add a title to the chart.
plt.title("Confusion Matrix")

# Display the chart.
plt.show()


# ---------------------------------------------------------
# 9. PRINT THE NUMERICAL EVALUATION METRICS
# ---------------------------------------------------------

print("Accuracy Measure:", accuracy)
print("F1 Score:", f1)