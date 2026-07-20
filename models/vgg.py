import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision import models


def build_model(model_name: str, num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, freezed: bool = False) -> nn.Module:
    """
    Constructs a VGG16 model with a regression head for age estimation.
    The model consists of convolutional blocks followed by fully connected layers.
    The regression head outputs a single value representing the estimated age.

    Args:
        input_size (int): Size of the input image (default: 224).
        num_classes (int): Number of classes for classification (default: 1000).
        num_channels (int): Number of input channels (default: 1 for grayscale images).
        num_outputs (int): Number of outputs for regression (default: 1).
        dropout_rate (float): Dropout rate for regularization (default: 0.3).

    Returns:
        nn.Module: A VGG model with a regression head.    
    """
    if model_name == "vgg11":
        model = models.vgg11(weights=None)
    elif model_name == "vgg13":
        model = models.vgg13(weights=None)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
    elif model_name == "vgg19":
        model = models.vgg19(weights=None)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models are 'vgg16' and 'vgg19'.")
    model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
    model.regression_head = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(4096, num_outputs),
        nn.SiLU()  # Use SiLU activation for regression output
    )

    if freezed:
        for param in model.features.parameters():
            param.requires_grad = False

    return model

def VGG11(num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, freezed: bool = False) -> nn.Module:
    return build_model("vgg11", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)

def VGG13(num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, freezed: bool = False) -> nn.Module:
    return build_model("vgg13", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)

def VGG16(num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, freezed: bool = False) -> nn.Module: 
    return build_model("vgg16", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)


def VGG19(num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, freezed: bool = False) -> nn.Module:
    return build_model("vgg19", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)


if __name__ == "__main__":
  vgg_16 = VGG19(num_channels=1, num_outputs=1, dropout_rate=0.3, pretrained=True, freezed=True)
  print(vgg_16)  # Print the model architecture
  # Create a random input tensor with shape (batch_size, channels, height, width)
  # For example, batch size = 16, channels = 1 (grayscale), height = 48, width = 48
  input_tensor = torch.rand(16, 1, 48, 48) # random imput tensor
  output = vgg_16(input_tensor)
  print("Output shape:", output.shape) # print the shape