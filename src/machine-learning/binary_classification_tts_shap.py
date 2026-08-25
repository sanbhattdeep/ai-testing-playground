import numpy as np
import shap

# Used to split the dataset into training and test sets
from sklearn.model_selection import train_test_split

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression

# Metrics used to evaluate the classifier
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# 1. CREATE SAMPLE DATA
# ---------------------------------------------------------

# Create 100 examples, each containing 2 input features.
#
# Shape of X:
#
#   100 rows    -> 100 examples
#   2 columns   -> Feature 1 and Feature 2
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


# Create 100 random binary labels.
#
# Each label is either:
#
#   0 -> Class 0
#   1 -> Class 1
#
# IMPORTANT:
# X and y are generated independently and randomly.
# Therefore, there is no meaningful relationship for
# the model to learn.
#
# This script is useful for learning SHAP mechanics,
# but the explanation itself should not be interpreted
# as meaningful real-world evidence.
y = np.random.randint(0, 2, 100)


# ---------------------------------------------------------
# 2. SPLIT DATA INTO TRAINING AND TEST SETS
# ---------------------------------------------------------

# Divide the data into:
#
# X_train -> features used for training
# X_test  -> features used for testing
# y_train -> correct labels used during training
# y_test  -> true labels used during evaluation
#
# test_size=0.2 means:
#
#   80% training
#   20% testing
#
# Since there are 100 examples:
#
#   80 training examples
#   20 test examples
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------------
# 3. CREATE AND TRAIN THE MODEL
# ---------------------------------------------------------

# Create a Logistic Regression binary classifier.
model = LogisticRegression()


# Train the model using the training data.
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. MAKE PREDICTIONS ON THE TEST SET
# ---------------------------------------------------------

# Predict either Class 0 or Class 1 for every test example.
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 5. EVALUATE THE MODEL
# ---------------------------------------------------------

# Accuracy:
# Fraction of all predictions that were correct.
accuracy = accuracy_score(y_test, y_pred)


# F1:
# Combines precision and recall into one metric.
f1 = f1_score(y_test, y_pred)


print("Accuracy Measure:", accuracy)
print("F1 Score:", f1)


# ---------------------------------------------------------
# 6. SELECT ONE TEST EXAMPLE TO EXPLAIN
# ---------------------------------------------------------

# SHAP can explain individual predictions.
#
# Here we choose the first example in the test set.
instance_index = 0


# Retrieve that example.
#
# Since X contains 2 features, instance might look like:
#
# [0.38, 0.25]
instance = X_test[instance_index]


# Print some useful information so we know exactly
# which prediction SHAP is explaining.
print("\nInstance being explained:", instance)
print("Actual class:", y_test[instance_index])
print("Predicted class:", model.predict([instance])[0])
print(
    "Predicted probabilities:",
    model.predict_proba([instance])[0]
)


# ---------------------------------------------------------
# 7. CREATE THE SHAP EXPLAINER
# ---------------------------------------------------------

# KernelExplainer is a model-agnostic SHAP explainer.
#
# It does not need to understand the internal mathematics
# of Logistic Regression.
#
# Instead, it repeatedly calls:
#
#     model.predict_proba(...)
#
# to observe how the model's output changes when features
# are present/absent or varied.
#
# X_train is used as the BACKGROUND dataset.
#
# The background data gives SHAP a reference for what
# "normal" feature values and normal model predictions
# look like.
explainer = shap.KernelExplainer(
    model.predict_proba,
    X_train
)


# ---------------------------------------------------------
# 8. CALCULATE SHAP VALUES
# ---------------------------------------------------------

# Pass the instance as one row.
#
# reshape(1, -1) changes:
#
#     [0.38, 0.25]
#
# into:
#
#     [[0.38, 0.25]]
#
# This makes the sample dimension explicit and avoids some
# ambiguity between SHAP versions.
instance_2d = instance.reshape(1, -1)


# Calculate SHAP values for this particular prediction.
#
# Because predict_proba() produces TWO outputs:
#
#     probability of Class 0
#     probability of Class 1
#
# SHAP calculates feature contributions for both outputs.
shap_values = explainer.shap_values(instance_2d)


# ---------------------------------------------------------
# 9. EXTRACT THE SHAP VALUES FOR CLASS 1
# ---------------------------------------------------------

# SHAP's return format differs between older and newer versions.
#
# Older versions may return a list:
#
#     [
#         values_for_class_0,
#         values_for_class_1
#     ]
#
# Newer versions may return an ndarray with shape:
#
#     (samples, features, outputs)
#
# The following handles both formats.
if isinstance(shap_values, list):
    # Older SHAP format:
    # shap_values[1] = contributions for Class 1
    class_1_shap_values = shap_values[1][0]

else:
    # Newer SHAP format:
    # sample 0, all features, output/class 1
    class_1_shap_values = shap_values[0, :, 1]


# expected_value is the model's BASELINE output.
#
# Since predict_proba() returns probabilities for both classes:
#
# expected_value[0] -> baseline probability for Class 0
# expected_value[1] -> baseline probability for Class 1
#
# We want to explain Class 1 here.
class_1_base_value = explainer.expected_value[1]

print("\n--- SHAP Explanation ---")

print("Class 1 base value:", class_1_base_value)

print(
    "Feature 1 SHAP value:",
    class_1_shap_values[0]
)

print(
    "Feature 2 SHAP value:",
    class_1_shap_values[1]
)

print(
    "Sum of SHAP values:",
    np.sum(class_1_shap_values)
)

print(
    "Base + SHAP values:",
    class_1_base_value + np.sum(class_1_shap_values)
)

print(
    "Model P(Class 1):",
    model.predict_proba([instance])[0][1]
)


# ---------------------------------------------------------
# 10. CREATE A SHAP FORCE PLOT
# ---------------------------------------------------------

# A force plot visually shows:
#
#     baseline prediction
#
#           +
#
#     Feature 1 contribution
#
#           +
#
#     Feature 2 contribution
#
#           =
#
#     model's prediction for this example
#
#
# Positive SHAP values push the Class-1 prediction upward.
# Negative SHAP values push the Class-1 prediction downward.
force_plot = shap.force_plot(
    class_1_base_value,
    class_1_shap_values,
    instance,
    feature_names=["Feature 1", "Feature 2"],
)


# ---------------------------------------------------------
# 11. SAVE THE EXPLANATION AS HTML
# ---------------------------------------------------------

# Save the interactive SHAP force plot as an HTML file.
#
# Open this file in a browser to inspect the explanation.
shap.save_html(
    "classify_explanation.html",
    force_plot
)