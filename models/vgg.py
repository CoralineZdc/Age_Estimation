import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision import models


def build_model(model_name: str, num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, pretrained: bool = False, freezed: bool = False) -> nn.Module:
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
        nn.Module: A VGG16 model with a regression head.    
    """
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
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


def VGG16(num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, pretrained: bool = False, freezed: bool = False) -> nn.Module: 
    return build_model("vgg16", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, pretrained=pretrained, freezed=freezed)


def VGG19(num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.3, pretrained: bool = False, freezed: bool = False) -> nn.Module:
    return build_model("vgg19", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, pretrained=pretrained, freezed=freezed)


"""
def VGG_conv_block(in_channels, out_channels, num_convs):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, input_size=224, num_classes=1000, num_channels=1, num_outputs=1, dropout_rate=0.3):
        super(VGG16, self).__init__()
        # Define convolutional blocks (features)
        self.input_size = input_size

        self.block1 = VGG_conv_block(num_channels, 64, 2)
        self.block2 = VGG_conv_block(64, 128, 2)
        self.block3 = VGG_conv_block(128, 256, 3)
        self.block4 = VGG_conv_block(256, 512, 3)
        self.block5 = VGG_conv_block(512, 512, 3)


        feature_dim = 512 * (input_size // 32) * (input_size // 32)
        # Fully connected layers (classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

        # Regression head for age estimation
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_outputs),
            nn.SiLU()  # Use SiLU activation for regression output
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # If x is 3D (C, H, W), add a batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
    
        # x = self.classifier(x)  # Uncomment this line if you want to use the classifier
        x = self.regression_head(x) # Uncomment this line if you want to use the regression head
        return x
    

class VGG19(nn.Module):
    def __init__(self, input_size, num_classes=1000, num_channels=1, num_outputs=1, dropout_rate=0.3):
        super(VGG19, self).__init__()
        self.input_size = input_size
        
        # Define convolutional blocks (features)
        self.block1 = VGG_conv_block(num_channels, 64, 2)
        self.block2 = VGG_conv_block(64, 128, 2)
        self.block3 = VGG_conv_block(128, 256, 4)
        self.block4 = VGG_conv_block(256, 512, 4)
        self.block5 = VGG_conv_block(512, 512, 4)

        feature_dim = 512 * (input_size // 32) * (input_size // 32)
        # Fully connected layers (classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

        # Regression head for age estimation
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_outputs),
            nn.SiLU()  # Use SiLU activation for regression output
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # If x is 3D (C, H, W), add a batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
    
        # x = self.classifier(x)  # Uncomment this line if you want to use the classifier
        x = self.regression_head(x) # Uncomment this line if you want to use the regression head
        return x
"""

if __name__ == "__main__":
  vgg_16 = VGG19(num_channels=1, num_outputs=1, dropout_rate=0.3, pretrained=True, freezed=True)
  print(vgg_16)  # Print the model architecture
  # Create a random input tensor with shape (batch_size, channels, height, width)
  # For example, batch size = 16, channels = 1 (grayscale), height = 48, width = 48
  input_tensor = torch.rand(16, 1, 48, 48) # random imput tensor
  output = vgg_16(input_tensor)
  print("Output shape:", output.shape) # print the shape