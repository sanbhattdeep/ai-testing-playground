import numpy as np
import matplotlib.pyplot as plt

# Used to split the dataset into training data and test data
from sklearn.model_selection import train_test_split

# Logistic Regression is a classification algorithm
from sklearn.linear_model import LogisticRegression

# Metrics used to evaluate the quality of predictions
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# 1. CREATE SAMPLE DATA
# ---------------------------------------------------------

# Create 100 examples (rows), each with 2 features (columns).
#
# Shape of X = (100, 2)
# This means:
#   - 100 data points / observations
#   - 2 input features for each data point
#
# Each value is a random decimal number between 0 and 1.
#
# Example row:
#   [0.42, 0.88]
#
X = np.random.rand(100, 2)


# Create 100 random labels, each either 0 or 1.
#
# This makes the problem a binary classification problem:
#   class 0 or class 1
#
# Example:
#   y = [0, 1, 0, 1, 1, 0, ...]
#
# IMPORTANT:
# Since X and y are generated randomly and independently,
# there is no real pattern for the model to learn.
# This script is useful for understanding the ML workflow,
# but the model performance is not expected to be meaningful.
y = np.random.randint(0, 2, 100)


# ---------------------------------------------------------
# 2. SPLIT THE DATA INTO TRAINING AND TEST SETS
# ---------------------------------------------------------

# Split the dataset into:
#   X_train -> features used to train the model
#   X_test  -> features used to test the model
#   y_train -> labels used to train the model
#   y_test  -> true labels used to evaluate the model
#
# test_size=0.2 means 20% of the data goes into the test set.
# Since we have 100 examples:
#   - 80 examples go to training
#   - 20 examples go to testing
#
# random_state=42 makes the split reproducible.
# This means the same split is used each time the script runs.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------------------------------------
# 3. CREATE AND TRAIN THE MODEL
# ---------------------------------------------------------

# Create a Logistic Regression classifier.
model = LogisticRegression()

# Train the model using the training data.
#
# The model tries to learn the relationship between:
#   input features (X_train)
# and
#   target labels (y_train)
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. MAKE PREDICTIONS ON THE TEST SET
# ---------------------------------------------------------

# Predict the class (0 or 1) for each example in X_test.
#
# y_pred will contain the model's predicted labels.
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 5. EVALUATE THE MODEL
# ---------------------------------------------------------

# Accuracy = fraction of predictions that are correct.
#
# Example:
# If 15 out of 20 predictions are correct:
#   accuracy = 15 / 20 = 0.75
accuracy = accuracy_score(y_test, y_pred)


# F1 score combines precision and recall into a single metric.
#
# It is especially useful when:
#   - classes are imbalanced
#   - false positives and false negatives both matter
#
# F1 ranges from 0 to 1:
#   1 = perfect
#   0 = very poor
f1 = f1_score(y_test, y_pred)


# ---------------------------------------------------------
# 6. VISUALIZE ACTUAL VS PREDICTED LABELS
# ---------------------------------------------------------

# First scatter plot:
# Plot the test points using their ACTUAL labels (y_test).
#
# X_test[:, 0] means:
#   all rows, column 0  -> Feature 1
#
# X_test[:, 1] means:
#   all rows, column 1  -> Feature 2
#
# c=y_test means the point colors are based on the true labels.
# cmap="bwr" means:
#   class 0 and class 1 will be shown using different colors
#   from the blue-white-red color map
#
# marker="s" means square markers
plt.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", label="Actual", marker="s"
)


# Second scatter plot:
# Plot the SAME test points again, but this time using the
# PREDICTED labels (y_pred).
#
# marker="o" means circle markers
# edgecolors="black" adds a black border around the circles
# linewidths=1 controls the thickness of that border
#
# Because the same points are plotted twice:
#   - squares represent actual labels
#   - circles represent predicted labels
#
# If a square and circle at the same location have the same color,
# the prediction matched the actual label.
#
# If they have different colors, the prediction was wrong.
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    cmap="bwr",
    label="Predicted",
    edgecolors="black",
    linewidths=1,
    marker="o",
)


# Add labels to the axes
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Add a title to the plot
plt.title("Binary Classification Results")

# Show the legend so we can distinguish Actual vs Predicted
plt.legend()

# Display the plot
plt.show()


# ---------------------------------------------------------
# 7. PRINT NUMERICAL EVALUATION METRICS
# ---------------------------------------------------------

print("Accuracy Measure:", accuracy)
print("F1 Score:", f1)