import torch
from torch import nn
from torchvision import models

class MobileNetClass(nn.Module):
    def __init__(self, num_outputs: int = 1, num_channels: int = 1, dropout_rate: float = 0.3):
        super(MobileNetClass, self).__init__()
        self.backbone = models.mobilenet_v2(weights=None)

        # Modify the first convolutional layer to accept the specified number of input channels
        self.backbone.features[0][0] = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Replace the classification head with a regression head for age estimation
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_features, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, num_outputs)  # Output: Age estimation (single value)
        )

    def forward(self, x):
        return self.backbone(x)
    

def MobileNet(num_outputs: int = 1, num_channels: int = 1, dropout_rate: float = 0.3, pretrained: bool = False, freezed: bool = False) -> nn.Module:
    model = MobileNetClass(num_outputs=num_outputs, num_channels=num_channels, dropout_rate=dropout_rate)
    if freezed:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True
    return model


if __name__ == "__main__":
    # Test with a dummy input
    model = MobileNet()
    dummy_input = torch.randn(4, 1, 48, 48) # Batch of size 4, 1 channel, 48x48
    print(f"Input shape: {dummy_input.shape}")
    print(model)  # Print the model architecture
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")