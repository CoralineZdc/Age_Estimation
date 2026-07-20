import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 

def build_model(model_name: str, num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.0, freezed: bool = False) -> nn.Module:
    """
    Constructs a ResNet model with a regression head for age estimation.
    The model consists of convolutional blocks followed by fully connected layers.
    The regression head outputs a single value representing the estimated age.

    Args:
        model_name (str): Name of the ResNet model to use (e.g., "resnet18").
        num_channels (int): Number of input channels (default: 1 for grayscale images).
        num_outputs (int): Number of outputs for regression (default: 1).
        dropout_rate (float): Dropout rate for regularization (default: 0.0).
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models are 'resnet18', 'resnet34', and 'resnet50'.")

    # Modify the first convolutional layer to accept the specified number of input channels
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.bn1 = nn.BatchNorm2d(64)

    # Replace the fully connected layer with a regression head
    in_features = model.fc.in_features
    hidden_features = 256
    out_features = num_outputs
    model.fc = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, out_features),
        )

    if freezed:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def ResNet18(num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.0, pretrained: bool = False, freezed: bool = False) -> nn.Module:
    return build_model("resnet18", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)


def ResNet34(num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.0, pretrained: bool = False, freezed: bool = False) -> nn.Module:
    return build_model("resnet34", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)


def ResNet50(num_channels: int = 1, num_outputs: int = 1, dropout_rate: float = 0.0, pretrained: bool = False, freezed: bool = False) -> nn.Module:
    return build_model("resnet50", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)

"""
class RegressionHead(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0, hidden_features=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetRegression(nn.Module):
    def __init__(self, block, num_blocks, num_outputs=1, dropout_rate=0.0, separate_heads=False):
        super(ResNetRegression, self).__init__()
        self.in_planes = 64
        self.separate_heads = separate_heads

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        out_features = 512 * block.expansion
        self.linear = RegressionHead(out_features, num_outputs, dropout_rate=dropout_rate)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)

        return self.linear(out)

def ResNet18(dropout_rate=0.0):
    # Backward-compatible alias
    return ResNetRegression(
        BasicBlock, 
        [2, 2, 2, 2], 
        num_outputs=1, 
        dropout_rate=dropout_rate
    )
"""




if __name__ == "__main__":
    # Test with ResNet18
    print("--- Testing ResNet18 ---")
    model18 = ResNet18(pretrained=True, freezed=True)
    dummy_input = torch.randn(4, 1, 48, 48)
    
    try:
        output18 = model18(dummy_input)
        print(f"ResNet18 Input: {dummy_input.shape} -> Output: {output18.shape}")
        print("Success!")
    except Exception as e:
        print(f"Error with ResNet18: {e}")

    # Test with ResNet50 for dynamic verification
    print("\n--- Testing ResNet50 ---")
    model50 = ResNet50(pretrained=False, freezed=False) # False for faster testing
    try:
        output50 = model50(dummy_input)
        print(f"ResNet50 Input: {dummy_input.shape} -> Output: {output50.shape}")
        print("Success!")
    except Exception as e:
        print(f"Error with ResNet50: {e}")