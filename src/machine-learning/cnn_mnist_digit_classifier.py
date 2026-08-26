import numpy as np

# Used here to split the original training dataset again
# into a training set and a validation set.
from sklearn.model_selection import train_test_split

# Scikit-learn metrics used to evaluate the final model
# on the test dataset.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Keras provides the MNIST dataset and the neural-network
# layers/model APIs used to build the CNN.
from tensorflow import keras


# ---------------------------------------------------------
# 1. LOAD THE MNIST HANDWRITTEN-DIGIT DATASET
# ---------------------------------------------------------

# MNIST is a dataset of handwritten digits from 0 through 9.
#
# Each image is:
#
#   28 pixels wide
#   28 pixels high
#   grayscale
#
# Keras already provides an official train/test split:
#
# Training:
#   X_train -> 60,000 images
#   y_train -> 60,000 digit labels
#
# Testing:
#   X_test  -> 10,000 images
#   y_test  -> 10,000 digit labels
#
# A label might look like:
#
#   7
#
# meaning that the corresponding image contains
# a handwritten number 7.
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# ---------------------------------------------------------
# 2. RESHAPE AND NORMALIZE THE IMAGE DATA
# ---------------------------------------------------------

# Initially an MNIST image has shape:
#
#     (28, 28)
#
# meaning:
#
#     28 rows of pixels
#     28 columns of pixels
#
# A Conv2D layer expects image data with an additional
# channel dimension:
#
#     (height, width, channels)
#
# Since MNIST images are grayscale, there is only 1 channel.
#
# Therefore:
#
#     (28, 28)
#
# becomes:
#
#     (28, 28, 1)
#
#
# -1 tells NumPy:
#
# "Automatically determine how many images there are."
#
# So the complete training shape changes approximately from:
#
#     (60000, 28, 28)
#
# to:
#
#     (60000, 28, 28, 1)
#
#
# Pixel values originally range from:
#
#     0 -> black
#     255 -> white
#
# Dividing by 255.0 normalizes them into:
#
#     0.0 -> 1.0
#
# Neural networks generally train more effectively when
# input values are on a smaller, consistent numerical scale.
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0

X_test = X_test.reshape(-1, 28, 28, 1) / 255.0


# ---------------------------------------------------------
# 3. CONVERT LABELS TO ONE-HOT ENCODING
# ---------------------------------------------------------

# Initially, a label is simply an integer.
#
# For example:
#
#     7
#
# Since this model has 10 possible output classes,
# Keras converts the label into a one-hot encoded vector.
#
# Example:
#
# Digit 0:
#
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
# Digit 3:
#
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#
# Digit 7:
#
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#
# num_classes=10 because MNIST contains digits:
#
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9
y_train = keras.utils.to_categorical(y_train, num_classes=10)

y_test = keras.utils.to_categorical(y_test, num_classes=10)


# ---------------------------------------------------------
# 4. CREATE A VALIDATION SET
# ---------------------------------------------------------

# Keras already gave us:
#
#     Training data
#     Test data
#
# We now split the training data again to create:
#
#     Training data
#     Validation data
#     Test data
#
# test_size=0.2 means 20% of the original training set
# becomes validation data.
#
# Original training set:
#
#     60,000 images
#
# After the split:
#
#     ~48,000 training images
#     ~12,000 validation images
#
# The original 10,000 test images remain untouched.
#
# random_state=42 makes this particular split reproducible.
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# ---------------------------------------------------------
# 5. BUILD THE CONVOLUTIONAL NEURAL NETWORK
# ---------------------------------------------------------

# Sequential means the layers are arranged one after another:
#
# Input image
#     ↓
# Conv2D
#     ↓
# MaxPooling
#     ↓
# Flatten
#     ↓
# Dense output layer
#     ↓
# Digit prediction
#
model = keras.models.Sequential(
    [
        # -------------------------------------------------
        # CONVOLUTIONAL LAYER
        # -------------------------------------------------
        #
        # Conv2D learns visual patterns from the images.
        #
        # 32:
        #   Learn 32 different filters.
        #
        # (3, 3):
        #   Each filter examines a 3×3 region of pixels
        #   at a time.
        #
        # activation="relu":
        #   Applies the ReLU activation function.
        #
        # input_shape=(28, 28, 1):
        #   Each input is a 28×28 grayscale image.
        #
        # During training, different filters may learn
        # useful visual patterns such as:
        #
        #   edges
        #   curves
        #   strokes
        #   corners
        #
        keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(28, 28, 1),
        ),

        # -------------------------------------------------
        # MAX-POOLING LAYER
        # -------------------------------------------------
        #
        # MaxPooling reduces the spatial dimensions of
        # the feature maps.
        #
        # (2, 2) means:
        #
        # Look at each 2×2 area and keep the largest value.
        #
        # This makes the representation smaller while
        # preserving strong detected features.
        keras.layers.MaxPooling2D((2, 2)),

        # -------------------------------------------------
        # FLATTEN LAYER
        # -------------------------------------------------
        #
        # The convolution/pooling layers produce
        # multi-dimensional feature maps.
        #
        # Flatten converts them into one long vector so
        # that they can be passed into the Dense layer.
        #
        # Conceptually:
        #
        # 3D feature maps
        #
        #     ↓
        #
        # [0.2, 0.8, 0.0, 0.7, ...]
        #
        keras.layers.Flatten(),

        # -------------------------------------------------
        # OUTPUT LAYER
        # -------------------------------------------------
        #
        # There are 10 output neurons because there are
        # 10 possible classes:
        #
        #     0 through 9
        #
        # softmax converts the 10 output values into
        # probabilities that sum to 1.
        #
        # Example:
        #
        # Digit 0 -> 0.01
        # Digit 1 -> 0.02
        # Digit 2 -> 0.03
        # Digit 3 -> 0.91
        # ...
        #
        # The largest probability becomes the prediction.
        keras.layers.Dense(10, activation="softmax"),
    ]
)


# ---------------------------------------------------------
# 6. CONFIGURE THE MODEL FOR TRAINING
# ---------------------------------------------------------

# compile() specifies HOW the neural network will learn.
#
# optimizer="adam":
#
# Adam updates the neural-network weights during training
# in an attempt to reduce the loss.
#
#
# loss="categorical_crossentropy":
#
# This measures how far the predicted probability
# distribution is from the correct one-hot encoded label.
#
# It is commonly used for multi-class classification when
# the labels have been one-hot encoded.
#
#
# metrics=["accuracy"]:
#
# Ask Keras to report accuracy during training and validation.
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# ---------------------------------------------------------
# 7. TRAIN THE MODEL
# ---------------------------------------------------------

# fit() trains the neural network.
#
# X_train:
#   Training images
#
# y_train:
#   Correct one-hot encoded digit labels
#
#
# batch_size=128:
#
# Instead of processing all ~48,000 images at once,
# the model processes groups of 128 images.
#
# After each batch, the model updates its weights.
#
#
# epochs=5:
#
# The model goes through the entire training dataset
# five times.
#
#
# validation_data=(X_val, y_val):
#
# After each epoch, Keras evaluates the model on the
# validation set.
#
# The validation set is NOT used for updating weights.
#
# It helps us observe whether performance on unseen data
# is improving or deteriorating.
model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=5,
    validation_data=(X_val, y_val),
)


# ---------------------------------------------------------
# 8. MAKE PREDICTIONS ON THE TEST DATA
# ---------------------------------------------------------

# predict() returns the model's predicted probabilities
# for every test image.
#
# For one image, the result might look like:
#
# [
#     0.001,
#     0.002,
#     0.003,
#     0.970,
#     0.004,
#     ...
# ]
#
# There are 10 probability values because there are
# 10 possible digits.
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 9. CONVERT PREDICTED PROBABILITIES TO CLASS NUMBERS
# ---------------------------------------------------------

# np.argmax(..., axis=1) finds the position containing
# the largest probability for each image.
#
# Example:
#
# probabilities:
#
# [0.01, 0.02, 0.05, 0.87, 0.01, ...]
#
# largest probability = 0.87
# position            = 3
#
# Therefore:
#
# predicted digit = 3
#
y_pred_classes = np.argmax(y_pred, axis=1)


# ---------------------------------------------------------
# 10. CONVERT TRUE ONE-HOT LABELS BACK TO DIGIT NUMBERS
# ---------------------------------------------------------

# y_test currently contains one-hot encoded values.
#
# Example:
#
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#
# np.argmax() converts this back to:
#
#     3
#
# This gives us ordinary integer labels that can be
# compared with y_pred_classes using sklearn metrics.
y_true_classes = np.argmax(y_test, axis=1)


# ---------------------------------------------------------
# 11. CALCULATE ACCURACY
# ---------------------------------------------------------

# Accuracy asks:
#
# "What fraction of all 10,000 test digits did the
# model classify correctly?"
#
# Formula:
#
#             correct predictions
# Accuracy = -----------------------
#             total predictions
#
accuracy = accuracy_score(
    y_true_classes,
    y_pred_classes,
)


# ---------------------------------------------------------
# 12. CALCULATE WEIGHTED PRECISION
# ---------------------------------------------------------

# Unlike the previous spam example, this is NOT binary
# classification.
#
# There are 10 classes:
#
#     0, 1, 2, ..., 9
#
# Precision can therefore be calculated separately
# for every digit.
#
# For example:
#
# Precision for digit 7:
#
# "Of everything predicted as 7,
#  how many were actually 7?"
#
#
# average="weighted" calculates precision for every class
# and then combines them, weighting each class according
# to how many actual examples of that class exist.
precision = precision_score(
    y_true_classes,
    y_pred_classes,
    average="weighted",
)


# ---------------------------------------------------------
# 13. CALCULATE WEIGHTED RECALL
# ---------------------------------------------------------

# Recall is calculated separately for every digit.
#
# Example for digit 7:
#
# "Of all the images that actually contained a 7,
#  how many did the model correctly recognize as 7?"
#
#
# average="weighted" combines the recall scores for all
# 10 classes, weighted by their number of examples.
recall = recall_score(
    y_true_classes,
    y_pred_classes,
    average="weighted",
)


# ---------------------------------------------------------
# 14. CALCULATE WEIGHTED F1 SCORE
# ---------------------------------------------------------

# F1 combines precision and recall.
#
# It is first determined for each of the 10 digit classes,
# and average="weighted" combines those class-level scores
# according to the number of actual examples in each class.
f1 = f1_score(
    y_true_classes,
    y_pred_classes,
    average="weighted",
)


# ---------------------------------------------------------
# 15. PRINT FINAL TEST METRICS
# ---------------------------------------------------------

print("Accuracy Measure:", accuracy)
print("Precision Measure:", precision)
print("Recall Measure:", recall)
print("F1 Score:", f1)