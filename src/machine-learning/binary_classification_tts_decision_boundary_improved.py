import numpy as np
import matplotlib.pyplot as plt

# Used to split the dataset into training and test sets
from sklearn.model_selection import train_test_split

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression

# Metrics used to evaluate the model
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# 1. CREATE SAMPLE DATA
# ---------------------------------------------------------

# Create 100 examples, each containing 2 input features.
#
# Shape of X:
#   100 rows    -> 100 examples
#   2 columns   -> Feature 1 and Feature 2
#
# Each value is a random decimal number between 0 and 1.
#
# Example:
# [
#     [0.25, 0.71],
#     [0.82, 0.13],
#     [0.44, 0.91],
#     ...
# ]
X = np.random.rand(100, 2)


# Create 100 random binary labels.
#
# Each label is either:
#   0 -> Class 0
#   1 -> Class 1
#
# IMPORTANT:
# X and y are generated independently, so there is no real
# relationship between the features and the labels.
#
# This is useful for learning how the ML workflow and decision
# boundary visualization work, but model performance will not
# be meaningful.
y = np.random.randint(0, 2, 100)


# ---------------------------------------------------------
# 2. SPLIT DATA INTO TRAINING AND TEST SETS
# ---------------------------------------------------------

# Split the dataset into:
#
# X_train -> features used to train the model
# X_test  -> features used to test the model
# y_train -> correct labels used during training
# y_test  -> correct labels used when evaluating the model
#
# test_size=0.2 means:
#   80% -> training
#   20% -> testing
#
# Since there are 100 examples:
#   80 training examples
#   20 test examples
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
model = LogisticRegression()


# Train the model using the training examples.
#
# The model learns a relationship between:
#
#   Feature 1 + Feature 2
#
# and:
#
#   Class 0 or Class 1
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. MAKE PREDICTIONS ON THE TEST SET
# ---------------------------------------------------------

# predict() returns the final predicted class:
#
#   0 or 1
#
# Since X_test contains 20 examples,
# y_pred will contain 20 predictions.
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 5. CALCULATE EVALUATION METRICS
# ---------------------------------------------------------

# Accuracy measures:
#
#   number of correct predictions
#   -----------------------------
#   total number of predictions
accuracy = accuracy_score(y_test, y_pred)


# F1 combines precision and recall into a single score.
#
# It is particularly useful when both false positives
# and false negatives matter.
f1 = f1_score(y_test, y_pred)


# ---------------------------------------------------------
# 6. CREATE A GRID OVER THE ENTIRE FEATURE SPACE
# ---------------------------------------------------------

# To visualize what the model would predict everywhere,
# not just for the 20 test points, create a grid covering:
#
# Feature 1: 0 -> 1
# Feature 2: 0 -> 1
#
# np.linspace(0, 1, 200) creates 200 evenly spaced values.
#
# Using 200 instead of 100 gives us a smoother visualization.
xx, yy = np.meshgrid(
    np.linspace(0, 1, 200),
    np.linspace(0, 1, 200)
)


# xx and yy are 2D arrays.
#
# The model expects input shaped like:
#
# [
#     [feature1, feature2],
#     [feature1, feature2],
#     ...
# ]
#
# ravel() flattens xx and yy.
#
# np.c_ combines the flattened values into coordinate pairs.
#
# Example:
#
# [
#     [0.00, 0.00],
#     [0.01, 0.00],
#     [0.02, 0.00],
#     ...
# ]
grid = np.c_[xx.ravel(), yy.ravel()]


# ---------------------------------------------------------
# 7. CALCULATE CLASS-1 PROBABILITIES FOR THE GRID
# ---------------------------------------------------------

# predict_proba() returns probabilities for BOTH classes.
#
# For one example it might return:
#
#     [0.80, 0.20]
#
# Meaning:
#
#     P(class 0) = 0.80
#     P(class 1) = 0.20
#
# Another example might return:
#
#     [0.25, 0.75]
#
# Meaning:
#
#     P(class 0) = 0.25
#     P(class 1) = 0.75
#
# [:, 1] selects only the probability of Class 1.
Z_prob = model.predict_proba(grid)[:, 1]


# Z_prob is currently a long 1D array.
#
# Reshape it back into the same 2D shape as xx and yy
# so Matplotlib can draw it across the feature space.
Z_prob = Z_prob.reshape(xx.shape)


# ---------------------------------------------------------
# 8. DRAW THE PROBABILITY REGIONS
# ---------------------------------------------------------

# contourf() fills the background based on the model's
# probability of Class 1.
#
# Instead of showing only:
#
#     blue -> Class 0
#     red  -> Class 1
#
# we now see the transition in probability between them.
#
# Conceptually:
#
#     Blue      -> low probability of Class 1
#     White-ish -> probability near 0.5
#     Red       -> high probability of Class 1
#
# levels=np.linspace(0, 1, 11) divides the probability
# range into intervals:
#
# 0.0, 0.1, 0.2, ..., 0.9, 1.0
plt.contourf(
    xx,
    yy,
    Z_prob,
    levels=np.linspace(0, 1, 11),
    alpha=0.8,
    cmap="bwr"
)


# ---------------------------------------------------------
# 9. EXPLICITLY DRAW THE DECISION BOUNDARY
# ---------------------------------------------------------

# Logistic Regression normally predicts:
#
#     Class 0 when P(class 1) < 0.5
#
#     Class 1 when P(class 1) >= 0.5
#
# Therefore:
#
#     P(class 1) = 0.5
#
# is the DECISION BOUNDARY.
#
# levels=[0.5] tells contour() to draw a line exactly
# where the predicted probability is 0.5.
#
# This black line is therefore the point where the model
# changes from predicting Class 0 to predicting Class 1.
plt.contour(
    xx,
    yy,
    Z_prob,
    levels=[0.5],
    colors="black",
    linewidths=3
)


# ---------------------------------------------------------
# 10. PLOT THE ACTUAL TEST DATA
# ---------------------------------------------------------

# Plot the 20 test examples on top of the probability map.
#
# X_test[:, 0] -> Feature 1
# X_test[:, 1] -> Feature 2
#
# c=y_test means:
#
#     Actual class 0 -> blue point
#     Actual class 1 -> red point
#
# Notice that the POINT COLOR represents the ACTUAL answer,
# while the BACKGROUND represents the model's prediction
# probability.
#
# edgecolors="black" adds a black outline around each point
# so the points remain visible against the background.
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap="bwr",
    edgecolors="black",
    linewidths=1,
    label="Actual"
)


# ---------------------------------------------------------
# 11. ADD THE COLOR BAR
# ---------------------------------------------------------

# The colorbar now has a meaningful continuous interpretation:
#
#     0.0 -> very low probability of Class 1
#     0.5 -> decision boundary
#     1.0 -> very high probability of Class 1
cbar = plt.colorbar()

cbar.set_label("Probability of Class 1")


# ---------------------------------------------------------
# 12. LABEL AND DISPLAY THE GRAPH
# ---------------------------------------------------------

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title(
    "Logistic Regression Results "
    "(Probability + Decision Boundary)"
)

plt.legend()

plt.show()


# ---------------------------------------------------------
# 13. PRINT MODEL PERFORMANCE
# ---------------------------------------------------------

print("Accuracy Measure:", accuracy)
print("F1 Score:", f1)