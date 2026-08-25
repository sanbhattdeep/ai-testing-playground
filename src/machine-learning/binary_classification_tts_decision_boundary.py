import numpy as np
import matplotlib.pyplot as plt

# Used to split data into training and test sets
from sklearn.model_selection import train_test_split

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression

# Metrics used to evaluate predictions
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# 1. CREATE SAMPLE DATA
# ---------------------------------------------------------

# Create 100 data points, each with 2 input features.
#
# Shape of X:
#   (100, 2)
#
# Meaning:
#   - 100 rows = 100 examples
#   - 2 columns = 2 features per example
#
# Each value is a random decimal number between 0 and 1.
X = np.random.rand(100, 2)


# Create 100 random labels, each either 0 or 1.
#
# This makes it a binary classification problem.
#
# IMPORTANT:
# X and y are random and unrelated here, so there is no real
# pattern for the model to learn. This is fine for learning the
# workflow, but the model's performance is not meaningful.
y = np.random.randint(0, 2, 100)


# ---------------------------------------------------------
# 2. SPLIT DATA INTO TRAINING AND TEST SETS
# ---------------------------------------------------------

# Split the dataset into:
#   X_train -> training features
#   X_test  -> test features
#   y_train -> training labels
#   y_test  -> true test labels
#
# test_size=0.2 means:
#   - 80% of the data goes to training
#   - 20% goes to testing
#
# Since we have 100 examples:
#   - 80 training examples
#   - 20 test examples
#
# random_state=42 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------------------------------------
# 3. CREATE AND TRAIN THE MODEL
# ---------------------------------------------------------

# Create a Logistic Regression classifier.
model = LogisticRegression()

# Train the model on the training data.
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. MAKE PREDICTIONS ON THE TEST SET
# ---------------------------------------------------------

# Predict class labels (0 or 1) for the test set.
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 5. EVALUATE THE MODEL
# ---------------------------------------------------------

# Accuracy = fraction of predictions that are correct.
accuracy = accuracy_score(y_test, y_pred)

# F1 score = harmonic mean of precision and recall.
f1 = f1_score(y_test, y_pred)


# ---------------------------------------------------------
# 6. CREATE A GRID OF POINTS FOR VISUALIZING THE
#    DECISION BOUNDARY
# ---------------------------------------------------------

# Create a grid of x and y values between 0 and 1.
#
# np.linspace(0, 1, 100) creates 100 evenly spaced values
# from 0 to 1.
#
# np.meshgrid() turns those x-values and y-values into
# coordinate matrices so we can evaluate the model across
# the whole 2D space.
#
# xx and yy each have shape (100, 100).
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))


# Convert the grid into a list of 2D points so the model
# can make predictions for every location in the grid.
#
# xx.ravel() flattens xx into a 1D array
# yy.ravel() flattens yy into a 1D array
#
# np.c_[xx.ravel(), yy.ravel()] combines them column-wise
# into pairs of coordinates:
#
# [
#   [x1, y1],
#   [x2, y2],
#   [x3, y3],
#   ...
# ]
#
# The model predicts class 0 or 1 for every grid point.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])


# Reshape the predictions back into the same 2D shape as xx and yy
# so they can be plotted as a contour map.
Z = Z.reshape(xx.shape)


# ---------------------------------------------------------
# 7. PLOT THE DECISION REGIONS AND TEST POINTS
# ---------------------------------------------------------

# Draw filled contour regions based on predicted class.
#
# This colors the background according to what class
# the model would predict in each area of the 2D space.
#
# alpha=0.8 makes the fill slightly transparent.
# cmap="bwr" uses a blue-white-red color map.
plt.contourf(xx, yy, Z, alpha=0.8, cmap="bwr")


# Plot the test data points on top of the background.
#
# X_test[:, 0] -> Feature 1 values (x-axis)
# X_test[:, 1] -> Feature 2 values (y-axis)
# c=y_test     -> color points using the ACTUAL labels
#
# So the background shows what the model predicts for each region,
# while the dots show the true labels of the test examples.
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", label="Actual")


# Add a colorbar to explain the background colors.
cbar = plt.colorbar()
cbar.set_label("Predicted")


# Label the axes and add a title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Binary Classification Results (Decision Boundary)")

# Show legend
plt.legend()

# Display the plot
plt.show()


# ---------------------------------------------------------
# 8. PRINT EVALUATION METRICS
# ---------------------------------------------------------

print("Accuracy Measure:", accuracy)
print("F1 Score:", f1)