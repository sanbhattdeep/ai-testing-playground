import numpy as np

# LIME (Local Interpretable Model-agnostic Explanations)
# is used to explain individual model predictions.
from lime import lime_tabular

# Used to split data into training and test sets.
from sklearn.model_selection import train_test_split

# Logistic Regression classifier.
from sklearn.linear_model import LogisticRegression

# Metrics used to evaluate model performance.
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# 1. CREATE SAMPLE DATA
# ---------------------------------------------------------

# Create 100 examples, each with 2 input features.
#
# Shape:
#   100 rows    -> 100 examples
#   2 columns   -> 2 features
#
# Each feature value is a random decimal between 0 and 1.
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
# X and y are generated independently, so there is no
# meaningful relationship between the features and labels.
#
# Therefore, this script is useful for learning LIME,
# but the resulting explanation should not be interpreted
# as a meaningful real-world explanation.
y = np.random.randint(0, 2, 100)


# ---------------------------------------------------------
# 2. SPLIT DATA INTO TRAINING AND TEST SETS
# ---------------------------------------------------------

# Divide the dataset into:
#
# X_train -> feature values used to train the model
# X_test  -> feature values used to test the model
# y_train -> labels used during training
# y_test  -> true labels used for evaluation
#
# test_size=0.2 means:
#
#   80% training data
#   20% test data
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
# 3. CREATE AND TRAIN THE CLASSIFICATION MODEL
# ---------------------------------------------------------

# Create a Logistic Regression classifier.
model = LogisticRegression()


# Train the model using the training data.
#
# The model attempts to learn:
#
# Feature 1 + Feature 2
#          ↓
#      Class 0 or 1
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. MAKE PREDICTIONS ON THE TEST DATA
# ---------------------------------------------------------

# Predict the class of every test example.
#
# y_pred might look like:
#
# [0, 1, 1, 0, 1, ...]
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 5. EVALUATE THE MODEL
# ---------------------------------------------------------

# Accuracy:
#
# How many predictions were correct out of
# all predictions made?
accuracy = accuracy_score(y_test, y_pred)


# F1 score:
#
# Combines precision and recall into one metric.
f1 = f1_score(y_test, y_pred)


print("Accuracy Measure:", accuracy)
print("F1 Score:", f1)


# ---------------------------------------------------------
# 6. SELECT ONE TEST EXAMPLE TO EXPLAIN
# ---------------------------------------------------------

# LIME explains INDIVIDUAL predictions.
#
# Here we select the first example from X_test.
#
# instance_index=0 means:
#
# "Explain the prediction for the first test example."
instance_index = 0


# Retrieve that particular test example.
#
# Since there are two features, instance might look like:
#
# [0.67, 0.24]
#
# where:
#
# Feature 1 = 0.67
# Feature 2 = 0.24
instance = X_test[instance_index]


# ---------------------------------------------------------
# 7. GIVE HUMAN-READABLE NAMES TO FEATURES AND CLASSES
# ---------------------------------------------------------

# Names of the two input features.
#
# These names will appear in the LIME explanation.
feature_names = [
    "Feature 1",
    "Feature 2"
]


# Names of the two possible output classes.
#
# Class index 0 -> "Class 0"
# Class index 1 -> "Class 1"
class_names = [
    "Class 0",
    "Class 1"
]


# ---------------------------------------------------------
# 8. CREATE THE LIME EXPLAINER
# ---------------------------------------------------------

# LimeTabularExplainer is used for tabular datasets
# such as:
#
# age | income | account_balance | ...
#
# LIME needs the training data so that it can understand
# the general distribution/range of the input features.
#
# We also provide human-readable feature and class names.
explainer = lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=class_names
)


# ---------------------------------------------------------
# 9. EXPLAIN THE SELECTED PREDICTION
# ---------------------------------------------------------

# explain_instance() asks LIME:
#
# "Why did the model make its prediction for THIS
# particular test example?"
#
# instance:
#   The specific example we want explained.
#
# model.predict_proba:
#   Function LIME uses to ask the model for probabilities.
#
# For example:
#
# model.predict_proba(...)
#
# might return:
#
# [0.25, 0.75]
#
# meaning:
#
# P(Class 0) = 25%
# P(Class 1) = 75%
#
#
# num_features=len(feature_names):
#
# We have 2 features, so LIME should include both
# features in the explanation.
explanation = explainer.explain_instance(
    instance,
    model.predict_proba,
    num_features=len(feature_names)
)


# ---------------------------------------------------------
# 10. SAVE THE LIME EXPLANATION AS AN HTML FILE
# ---------------------------------------------------------

# Save the interactive LIME explanation as an HTML file.
#
# The generated file can be opened in a browser.
#
# It typically shows:
#
# - predicted probabilities
# - which class was predicted
# - which features pushed the prediction toward Class 0
# - which features pushed the prediction toward Class 1
# - the feature values of the selected example
explanation.save_to_file(
    "classify_explanation.html"
)