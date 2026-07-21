import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet 

class EfficientNetClass(nn.Module):
    def __init__(self, in_channels: int =1, dropout_rate: float =0.3):
        super(EfficientNetClass, self).__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b0', in_channels=in_channels)  # Assuming grayscale images; change to 3 for RGB

        # Replace the classification head with a regression head for age estimation
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity() # Remove the original FC layer
        
        # Nouvelle tête de régression
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 128),
            nn.SiLU(), 
            nn.Linear(128, 1) # Output: Age estimation (single value)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.regression_head(features)
    
def EfficientNetModel(in_channels: int =1, dropout_rate: float = 0.3, freezed: bool =False) -> nn.Module:
    model = EfficientNetClass(in_channels=in_channels, dropout_rate=dropout_rate)

    if freezed:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.regression_head.parameters():
            param.requires_grad = True

    return model

if __name__ == "__main__":
    # Test with a dummy input
    model = EfficientNetModel()
    print(model)  # Print the model architecture
    dummy_input = torch.randn(4, 1, 48, 48) # Batch 0f size 4, 3 channes, 48x48
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")