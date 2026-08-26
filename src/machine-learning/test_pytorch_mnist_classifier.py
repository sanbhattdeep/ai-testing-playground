# Pytest is the testing framework used to define and execute
# automated tests.
import pytest

# PyTorch is used for tensors and loading the trained model.
import torch

# torch.nn provides neural-network building blocks such as
# Conv2d, MaxPool2d, Flatten, and Linear.
import torch.nn as nn

# MNIST contains handwritten digit images from 0 through 9.
from torchvision.datasets import MNIST

# ToTensor converts images into PyTorch tensors when images
# are retrieved through the Dataset interface.
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. LOAD THE MNIST TEST DATASET
# ---------------------------------------------------------

# Load only the official MNIST TEST dataset.
#
# root="data":
#   Store/read the dataset from the local "data" directory.
#
# train=False:
#   Load the 10,000 official TEST images rather than
#   the 60,000 training images.
#
# download=True:
#   Download the dataset if it is not already available.
#
# transform=ToTensor():
#   Normally converts an image to a tensor when accessed
#   using something such as:
#
#       mnist_test[index]
#
# IMPROVEMENT NOTE:
# This script directly accesses mnist_test.data below,
# so the ToTensor transform is not actually used by that path.
#
# The original script is left unchanged.
mnist_test = MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# ---------------------------------------------------------
# 2. PREPARE THE TEST IMAGES
# ---------------------------------------------------------

# mnist_test.data initially has shape:
#
#     (10000, 28, 28)
#
# meaning:
#
#     10,000 images
#     28 pixels high
#     28 pixels wide
#
# PyTorch Conv2d expects:
#
#     (samples, channels, height, width)
#
# MNIST is grayscale, so there is one channel.
#
# unsqueeze(1) changes:
#
#     (10000, 28, 28)
#
# into:
#
#     (10000, 1, 28, 28)
#
# .float() converts pixel values to floating point.
#
# Dividing by 255.0 normalizes pixel values from:
#
#     0–255
#
# into:
#
#     0.0–1.0
X_test = mnist_test.data.unsqueeze(1).float() / 255.0


# ---------------------------------------------------------
# 3. PREPARE THE TEST LABELS
# ---------------------------------------------------------

# MNIST labels originally look like:
#
#     0
#     5
#     7
#     9
#
# one_hot() converts each integer class into a vector.
#
# Example:
#
# Label 3:
#
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#
# Label 7:
#
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#
# .float() converts the tensor to floating point.
#
# IMPROVEMENT NOTE:
# One-hot encoding is not necessary for this test because
# the original integer labels in mnist_test.targets could
# be compared directly with the predicted class.
#
# The original script is left unchanged.
y_test = torch.nn.functional.one_hot(
    mnist_test.targets,
    num_classes=10
).float()


# ---------------------------------------------------------
# 4. DEFINE THE SAME CNN ARCHITECTURE USED DURING TRAINING
# ---------------------------------------------------------

# This class MUST match the architecture used when model.pth
# was created.
#
# model.pth contains the learned parameters, but not this
# Python class definition itself.
#
# Therefore, before loading the saved weights, we recreate
# the same network structure.
class SimpleCNN(nn.Module):

    def __init__(self):

        # Initialize PyTorch's base nn.Module class.
        super(SimpleCNN, self).__init__()


        # -------------------------------------------------
        # CONVOLUTIONAL LAYER
        # -------------------------------------------------

        # Input channels = 1 because MNIST is grayscale.
        #
        # Output channels = 32 because the network learns
        # 32 different convolution filters.
        #
        # kernel_size=3 means each filter is 3 × 3 pixels.
        #
        # stride=1 means the filter moves one pixel at a time.
        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            stride=1
        )


        # -------------------------------------------------
        # MAX-POOLING LAYER
        # -------------------------------------------------

        # Reduce the feature-map dimensions by examining
        # 2 × 2 areas and keeping the largest activation.
        self.maxpool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # -------------------------------------------------
        # FLATTEN LAYER
        # -------------------------------------------------

        # Convert the multi-dimensional feature maps into
        # one long vector before the fully connected layer.
        self.flatten = nn.Flatten()


        # -------------------------------------------------
        # FULLY CONNECTED OUTPUT LAYER
        # -------------------------------------------------

        # Input image:
        #
        #     28 × 28
        #
        # After the 3 × 3 convolution:
        #
        #     26 × 26 × 32
        #
        # After 2 × 2 max pooling:
        #
        #     13 × 13 × 32
        #
        # Flatten:
        #
        #     13 × 13 × 32
        #     = 5408 values
        #
        # Output:
        #
        #     10 logits
        #
        # one for each digit:
        #
        #     0 through 9
        self.fc = nn.Linear(
            13 * 13 * 32,
            10
        )


    # -----------------------------------------------------
    # 5. DEFINE HOW INPUT FLOWS THROUGH THE CNN
    # -----------------------------------------------------

    def forward(self, x):

        # Apply convolution.
        x = self.conv1(x)

        # Apply ReLU activation.
        #
        # Negative values become 0.
        # Positive values remain positive.
        x = nn.ReLU()(x)

        # Reduce spatial dimensions.
        x = self.maxpool(x)

        # Convert feature maps into a 1D feature vector
        # for each input image.
        x = self.flatten(x)

        # Produce 10 raw output scores (logits).
        x = self.fc(x)

        # Return the logits.
        return x


# ---------------------------------------------------------
# 6. CREATE AN EMPTY MODEL INSTANCE
# ---------------------------------------------------------

# At this point SimpleCNN has the correct architecture,
# but its weights are newly initialized/random.
model = SimpleCNN()


# ---------------------------------------------------------
# 7. LOAD THE PREVIOUSLY TRAINED WEIGHTS
# ---------------------------------------------------------

# torch.load("model.pth") reads the state dictionary
# saved by the training script.
#
# The previous script saved it using:
#
#     torch.save(model.state_dict(), "model.pth")
#
# load_state_dict() puts those learned weights and biases
# into this new SimpleCNN instance.
#
# So conceptually:
#
# SimpleCNN architecture
#         +
# model.pth weights
#         =
# trained classifier
model.load_state_dict(
    torch.load("model.pth")
)


# IMPROVEMENT NOTE:
# model.pth must exist relative to the directory from which
# pytest is executed.
#
# If pytest is run from a different working directory,
# this relative path could fail.
#
# In a larger project you might build the path relative
# to the test file instead.


# ---------------------------------------------------------
# 8. DEFINE A PARAMETERIZED PYTEST TEST
# ---------------------------------------------------------

# @pytest.mark.parametrize allows the SAME test function
# to run multiple times using different input values.
#
# Here:
#
#     index = 0
#     index = 1
#     index = 2
#     index = 3
#     index = 4
#
# Therefore pytest creates FIVE separate test cases.
#
# Conceptually:
#
# test_mnist_classification(0)
# test_mnist_classification(1)
# test_mnist_classification(2)
# test_mnist_classification(3)
# test_mnist_classification(4)
#
# Each one will be reported separately by pytest.
@pytest.mark.parametrize(
    "index",
    [0, 1, 2, 3, 4]
)
def test_mnist_classification(index):


    # -----------------------------------------------------
    # 9. PUT THE MODEL INTO EVALUATION MODE
    # -----------------------------------------------------

    # model.eval() switches the network into inference mode.
    #
    # This matters especially for layers such as:
    #
    #     Dropout
    #     BatchNorm
    #
    # This particular CNN doesn't contain those layers,
    # but calling model.eval() before inference is
    # standard PyTorch practice.
    model.eval()


    # -----------------------------------------------------
    # 10. SELECT ONE MNIST TEST IMAGE
    # -----------------------------------------------------

    # Select the image corresponding to this test index.
    #
    # X_test has shape:
    #
    #     (10000, 1, 28, 28)
    #
    # After selecting one image:
    #
    #     input_data.shape
    #
    # becomes:
    #
    #     (1, 28, 28)
    input_data = X_test[index]


    # -----------------------------------------------------
    # 11. GET THE EXPECTED / TRUE LABEL
    # -----------------------------------------------------

    # y_test[index] contains a one-hot encoded label.
    #
    # Example:
    #
    # [0,0,0,0,0,0,0,1,0,0]
    #
    # torch.argmax() returns the position of the 1:
    #
    #     7
    #
    # So "target" is the correct digit that the model
    # should predict.
    target = torch.argmax(
        y_test[index]
    )


    # -----------------------------------------------------
    # 12. DISABLE GRADIENT CALCULATION DURING INFERENCE
    # -----------------------------------------------------

    # During testing we are NOT training the model.
    #
    # Therefore we do not need:
    #
    #     gradients
    #     backpropagation
    #     optimizer updates
    #
    # torch.no_grad() tells PyTorch not to construct the
    # computation graph required for backpropagation.
    #
    # Benefits include:
    #
    #     less memory usage
    #     less computation
    with torch.no_grad():


        # -------------------------------------------------
        # 13. ADD THE BATCH DIMENSION
        # -------------------------------------------------

        # input_data currently has shape:
        #
        #     (1, 28, 28)
        #
        # But Conv2d expects:
        #
        #     (batch, channels, height, width)
        #
        # unsqueeze(0) adds the batch dimension:
        #
        #     (1, 28, 28)
        #
        # becomes:
        #
        #     (1, 1, 28, 28)
        #
        # Meaning:
        #
        #     1 image
        #     1 grayscale channel
        #     28 height
        #     28 width
        output = model(
            input_data.unsqueeze(0)
        )


        # -------------------------------------------------
        # 14. CONVERT MODEL OUTPUT INTO A DIGIT
        # -------------------------------------------------

        # The model produces 10 logits:
        #
        # one for each digit 0–9.
        #
        # Example:
        #
        # [
        #     -1.2,
        #      0.5,
        #     -0.3,
        #      4.8,
        #      0.7,
        #      ...
        # ]
        #
        # torch.argmax() finds the position of the
        # largest output.
        #
        # If index 3 contains the highest value:
        #
        #     predicted digit = 3
        predicted = torch.argmax(output)


    # -----------------------------------------------------
    # 15. ASSERT THAT THE PREDICTION IS CORRECT
    # -----------------------------------------------------

    # pytest considers the test successful if:
    #
    #     predicted == target
    #
    # Example:
    #
    # predicted = 7
    # target    = 7
    #
    #     PASS
    #
    #
    # If:
    #
    # predicted = 3
    # target    = 7
    #
    #     FAIL
    #
    # This is the actual automated test condition.
    assert predicted == target, f"Test failed for index {index}"

    image = X_test[index].squeeze().numpy()
    predicted_label = predicted.item()
    target_label = target.item()
 
    plt.imshow(image, cmap="gray")
    plt.title(f"Predicted: {predicted_label}, Target: {target_label}")
    plt.axis("off")
    plt.show()