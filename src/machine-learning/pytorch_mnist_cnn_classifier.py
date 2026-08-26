# Used to split the original MNIST training dataset
# into training and validation sets.
from sklearn.model_selection import train_test_split

# Scikit-learn metrics used to evaluate the final
# classification performance on the test dataset.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# PyTorch is the main deep-learning framework used here.
import torch

# torch.nn contains neural-network building blocks such as:
#
# Conv2d
# Linear
# ReLU
# MaxPool2d
# CrossEntropyLoss
import torch.nn as nn

# torch.optim contains optimization algorithms used to
# update the neural-network parameters during training.
import torch.optim as optim

# torchvision provides commonly used computer-vision datasets.
#
# MNIST contains images of handwritten digits 0 through 9.
from torchvision.datasets import MNIST

# ToTensor converts images into PyTorch tensors.
from torchvision.transforms import ToTensor


# ---------------------------------------------------------
# 1. LOAD THE MNIST DATASET
# ---------------------------------------------------------

# Download/load the MNIST TRAINING dataset.
#
# root="data":
#   Store the dataset inside a local "data" directory.
#
# train=True:
#   Load the 60,000 training examples.
#
# download=True:
#   Download MNIST if it does not already exist locally.
#
# transform=ToTensor():
#   Specifies that images retrieved normally from this
#   Dataset should be converted into PyTorch tensors.
#
# IMPROVEMENT NOTE:
# In this script we later access mnist_train.data directly,
# rather than retrieving items through mnist_train[index].
#
# Therefore ToTensor() is not actually used by the later
# code path. It would matter if you accessed the dataset
# through its normal Dataset interface.
mnist_train = MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


# Load the official MNIST TEST dataset.
#
# train=False gives the separate 10,000-image test set.
mnist_test = MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# ---------------------------------------------------------
# 2. PREPARE THE IMAGE DATA
# ---------------------------------------------------------

# mnist_train.data initially has approximately this shape:
#
#     (60000, 28, 28)
#
# Meaning:
#
#     60,000 images
#     28 pixels high
#     28 pixels wide
#
#
# A Conv2d layer expects data shaped like:
#
#     (batch, channels, height, width)
#
# PyTorch uses channel-first ordering.
#
# Since MNIST is grayscale:
#
#     channels = 1
#
#
# unsqueeze(1) inserts that channel dimension:
#
#     (60000, 28, 28)
#
# becomes:
#
#     (60000, 1, 28, 28)
#
#
# .float() converts the pixel values to floating point.
#
# Original pixel values range from:
#
#     0 to 255
#
# Dividing by 255.0 normalizes them to:
#
#     0.0 to 1.0
X_train = mnist_train.data.unsqueeze(1).float() / 255.0

X_test = mnist_test.data.unsqueeze(1).float() / 255.0


# ---------------------------------------------------------
# 3. ONE-HOT ENCODE THE LABELS
# ---------------------------------------------------------

# Original MNIST targets are integers:
#
#     0, 1, 2, ..., 9
#
# one_hot() converts them into vectors containing 10 values.
#
# Example:
#
# Digit 3:
#
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#
# Digit 7:
#
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#
#
# .float() converts the resulting tensor to floating point.
#
# IMPROVEMENT NOTE:
# PyTorch's CrossEntropyLoss normally expects integer
# class labels such as:
#
#     3
#     7
#     9
#
# rather than one-hot vectors.
#
# This script later converts the one-hot labels back to
# integers with torch.argmax(), so the one-hot encoding
# is not strictly necessary.
#
# The original script is left unchanged.
y_train = torch.nn.functional.one_hot(
    mnist_train.targets,
    num_classes=10,
).float()

y_test = torch.nn.functional.one_hot(
    mnist_test.targets,
    num_classes=10,
).float()


# ---------------------------------------------------------
# 4. CREATE A VALIDATION SET
# ---------------------------------------------------------

# The official MNIST dataset already gives us:
#
#     Training set -> 60,000 examples
#     Test set     -> 10,000 examples
#
# We split the 60,000 training examples again:
#
#     ~48,000 training
#     ~12,000 validation
#
# test_size=0.2 means 20% becomes validation data.
#
# random_state=42 makes this split reproducible.
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
)


# IMPROVEMENT NOTE:
# X_val and y_val are created here but are not actually
# used later in the script.
#
# In a typical neural-network workflow, you would evaluate
# validation loss/accuracy after every epoch to detect
# things such as overfitting.
#
# The original script is intentionally left unchanged.


# ---------------------------------------------------------
# 5. DEFINE THE CNN ARCHITECTURE
# ---------------------------------------------------------

# In PyTorch, neural networks are commonly created by
# defining a Python class that inherits from nn.Module.
#
# The class has two especially important methods:
#
#     __init__()
#
# Defines WHICH layers exist.
#
#     forward()
#
# Defines HOW data flows through those layers.
class SimpleCNN(nn.Module):

    def __init__(self):

        # Initialize the parent nn.Module class.
        super(SimpleCNN, self).__init__()


        # -------------------------------------------------
        # CONVOLUTIONAL LAYER
        # -------------------------------------------------

        # nn.Conv2d(
        #     1,
        #     32,
        #     kernel_size=3,
        #     stride=1
        # )
        #
        # 1:
        #   Number of input channels.
        #
        #   MNIST is grayscale, so there is 1 channel.
        #
        # 32:
        #   Number of filters/output feature maps.
        #
        # kernel_size=3:
        #   Each filter is 3 × 3 pixels.
        #
        # stride=1:
        #   Move the filter one pixel at a time.
        #
        # These filters can learn patterns such as:
        #
        #     edges
        #     curves
        #     strokes
        #     corners
        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            stride=1,
        )


        # -------------------------------------------------
        # MAX-POOLING LAYER
        # -------------------------------------------------

        # MaxPool2d reduces the spatial dimensions.
        #
        # kernel_size=2:
        #   Examine 2 × 2 regions.
        #
        # stride=2:
        #   Move two pixels at a time.
        #
        # For each 2 × 2 region, only the largest
        # activation is retained.
        self.maxpool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )


        # -------------------------------------------------
        # FLATTEN LAYER
        # -------------------------------------------------

        # Converts the multi-dimensional feature maps into
        # one long vector before passing them to a
        # fully connected layer.
        self.flatten = nn.Flatten()


        # -------------------------------------------------
        # FULLY CONNECTED OUTPUT LAYER
        # -------------------------------------------------

        # Let's calculate where 13 * 13 * 32 comes from.
        #
        # Original image:
        #
        #     28 × 28
        #
        # Conv2D with 3 × 3 kernel and no padding:
        #
        #     28 - 3 + 1
        #     = 26
        #
        # so:
        #
        #     26 × 26 × 32
        #
        # Max pooling with 2 × 2:
        #
        #     26 / 2 = 13
        #
        # giving:
        #
        #     13 × 13 × 32
        #
        # Flatten:
        #
        #     13 × 13 × 32
        #     = 5408 values
        #
        # The Dense/Linear layer converts those 5408
        # features into 10 outputs:
        #
        #     digit 0
        #     digit 1
        #     ...
        #     digit 9
        self.fc = nn.Linear(
            13 * 13 * 32,
            10,
        )


    # -----------------------------------------------------
    # 6. DEFINE THE FORWARD PASS
    # -----------------------------------------------------

    # forward() describes exactly how an input tensor moves
    # through the neural network.
    def forward(self, x):

        # Apply convolution.
        #
        # Input:
        #
        #     batch × 1 × 28 × 28
        #
        # Output:
        #
        #     batch × 32 × 26 × 26
        x = self.conv1(x)


        # Apply ReLU activation.
        #
        # ReLU(x) = max(0, x)
        #
        # Negative values become zero.
        # Positive values remain positive.
        x = nn.ReLU()(x)


        # Reduce each feature map using max pooling.
        #
        # Shape approximately becomes:
        #
        #     batch × 32 × 13 × 13
        x = self.maxpool(x)


        # Convert the feature maps into one long vector
        # for every image.
        #
        # Shape becomes:
        #
        #     batch × 5408
        x = self.flatten(x)


        # Produce 10 output values.
        #
        # IMPORTANT:
        # These values are called LOGITS.
        #
        # They are NOT probabilities yet.
        #
        # Example:
        #
        # [0.2, -1.1, 3.8, 0.4, ...]
        #
        # CrossEntropyLoss knows how to work directly
        # with logits.
        x = self.fc(x)


        # Return the logits.
        return x


# ---------------------------------------------------------
# 7. CREATE AN INSTANCE OF THE MODEL
# ---------------------------------------------------------

# At this point the architecture exists, but it has not
# learned anything yet.
model = SimpleCNN()


# ---------------------------------------------------------
# 8. DEFINE THE LOSS FUNCTION
# ---------------------------------------------------------

# CrossEntropyLoss is commonly used for multi-class
# classification.
#
# We have 10 possible classes:
#
#     0 through 9
#
# It compares the model's output logits with the
# correct class number.
#
# IMPORTANT:
# You should NOT manually apply softmax before
# CrossEntropyLoss.
#
# CrossEntropyLoss internally handles the appropriate
# log-softmax calculation.
criterion = nn.CrossEntropyLoss()


# ---------------------------------------------------------
# 9. DEFINE THE OPTIMIZER
# ---------------------------------------------------------

# Adam is used to update the neural-network parameters.
#
# model.parameters():
#   All trainable weights and biases in the CNN.
#
# lr=0.001:
#   Learning rate.
#
# The learning rate controls how large each weight update is.
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)


# ---------------------------------------------------------
# 10. DEFINE TRAINING SETTINGS
# ---------------------------------------------------------

# Train for five complete passes through the training data.
num_epochs = 5


# Process 128 images at a time.
batch_size = 128


# ---------------------------------------------------------
# 11. TRAIN THE MODEL
# ---------------------------------------------------------

# Repeat the training process for each epoch.
for epoch in range(num_epochs):


    # Put the model into TRAINING mode.
    #
    # This is important for layers whose behavior changes
    # between training and evaluation, such as:
    #
    #     Dropout
    #     BatchNorm
    #
    # This particular network doesn't contain those layers,
    # but calling model.train() is still good PyTorch practice.
    model.train()


    # Manually create batches by moving through X_train
    # in steps of 128 examples.
    #
    # Example:
    #
    # i = 0
    #     examples 0-127
    #
    # i = 128
    #     examples 128-255
    #
    # etc.
    for i in range(0, len(X_train), batch_size):


        # Select one batch of input images.
        inputs = X_train[i : i + batch_size]


        # y_train contains one-hot encoded labels.
        #
        # Example:
        #
        # [0,0,0,1,0,0,0,0,0,0]
        #
        # torch.argmax(..., dim=1) converts them back into
        # integer class labels:
        #
        #     3
        #
        # CrossEntropyLoss expects these integer class indexes.
        targets = torch.argmax(
            y_train[i : i + batch_size],
            dim=1,
        )


        # -------------------------------------------------
        # STEP 1: CLEAR OLD GRADIENTS
        # -------------------------------------------------

        # PyTorch accumulates gradients by default.
        #
        # Therefore we must clear gradients from the
        # previous batch before calculating new ones.
        optimizer.zero_grad()


        # -------------------------------------------------
        # STEP 2: FORWARD PASS
        # -------------------------------------------------

        # Pass the batch through the CNN.
        #
        # Internally this calls:
        #
        #     model.forward(inputs)
        #
        # outputs contains 10 logits per image.
        outputs = model(inputs)


        # -------------------------------------------------
        # STEP 3: CALCULATE LOSS
        # -------------------------------------------------

        # Compare model outputs against the correct labels.
        #
        # A higher loss means predictions are worse.
        # Training attempts to reduce this value.
        loss = criterion(
            outputs,
            targets,
        )


        # -------------------------------------------------
        # STEP 4: BACKPROPAGATION
        # -------------------------------------------------

        # Calculate gradients for every trainable parameter.
        #
        # Conceptually:
        #
        # "How should each weight change to reduce the loss?"
        loss.backward()


        # -------------------------------------------------
        # STEP 5: UPDATE MODEL WEIGHTS
        # -------------------------------------------------

        # Adam uses the calculated gradients to update
        # the model's parameters.
        optimizer.step()


# IMPROVEMENT NOTE:
# This script manually slices X_train to create batches.
#
# In typical PyTorch projects you would usually use:
#
#     torch.utils.data.DataLoader
#
# A DataLoader can:
#
#     create batches
#     shuffle training examples
#     use multiple workers
#     load data efficiently
#
# This script also uses the same training-data ordering
# each epoch rather than explicitly shuffling batches.
#
# The original code is left unchanged.


# ---------------------------------------------------------
# 12. SAVE THE TRAINED MODEL PARAMETERS
# ---------------------------------------------------------

# state_dict() contains the learned weights and biases.
#
# torch.save() writes those parameters to disk.
#
# The resulting file:
#
#     model.pth
#
# can later be loaded into another SimpleCNN instance.
torch.save(
    model.state_dict(),
    "model.pth",
)


# ---------------------------------------------------------
# 13. SWITCH THE MODEL TO EVALUATION MODE
# ---------------------------------------------------------

# model.eval() tells PyTorch that we are no longer training.
#
# This affects layers such as:
#
#     Dropout
#     BatchNorm
#
# Again, this model doesn't contain them, but using eval()
# before inference is standard PyTorch practice.
model.eval()


# ---------------------------------------------------------
# 14. PREDICT THE TEST CLASSES
# ---------------------------------------------------------

# model(X_test) runs all 10,000 test images through the CNN.
#
# The result contains 10 logits for every image.
#
# Example for one image:
#
#     [-1.2, 0.3, 0.5, 4.8, ...]
#
# torch.argmax(..., dim=1) selects the output having the
# largest value.
#
# If index 3 has the highest logit:
#
#     predicted digit = 3
y_pred = torch.argmax(
    model(X_test),
    dim=1,
)


# IMPROVEMENT NOTE:
# During evaluation, a typical PyTorch script would use:
#
#     with torch.no_grad():
#
# around inference.
#
# This tells PyTorch that gradients do not need to be tracked,
# which saves memory and computation.
#
# The original script is left unchanged.


# ---------------------------------------------------------
# 15. CONVERT TRUE LABELS BACK TO DIGIT NUMBERS
# ---------------------------------------------------------

# y_test currently contains one-hot encoded labels.
#
# Example:
#
#     [0,0,0,0,0,0,0,1,0,0]
#
# argmax converts that back into:
#
#     7
y_true = torch.argmax(
    y_test,
    dim=1,
)


# ---------------------------------------------------------
# 16. CALCULATE TEST ACCURACY
# ---------------------------------------------------------

# Accuracy asks:
#
# "Of all 10,000 test images, what fraction were
# classified correctly?"
accuracy = accuracy_score(
    y_true,
    y_pred,
)


# ---------------------------------------------------------
# 17. CALCULATE WEIGHTED PRECISION
# ---------------------------------------------------------

# Precision is calculated separately for each digit.
#
# Example for digit 7:
#
# "Of all images predicted as 7,
#  how many really were 7?"
#
# average="weighted" combines precision across all
# 10 classes, weighted by the number of actual examples
# belonging to each class.
precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
)


# ---------------------------------------------------------
# 18. CALCULATE WEIGHTED RECALL
# ---------------------------------------------------------

# Recall is also calculated separately for each digit.
#
# Example for digit 7:
#
# "Of all images that actually were 7,
#  how many did the model successfully recognize?"
#
# Weighted recall combines these class-level values
# according to class size.
recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
)


# ---------------------------------------------------------
# 19. CALCULATE WEIGHTED F1 SCORE
# ---------------------------------------------------------

# F1 combines precision and recall.
#
# F1 is calculated for each digit and then weighted
# according to how many actual examples of each digit exist.
f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
)


# ---------------------------------------------------------
# 20. PRINT FINAL TEST METRICS
# ---------------------------------------------------------

print("Accuracy Measure:", accuracy)
print("Precision Measure:", precision)
print("Recall Measure:", recall)
print("F1 Score:", f1)