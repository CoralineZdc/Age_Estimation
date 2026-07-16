import torch
import torch.nn.functional as F
from torch import nn

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, bias=bias)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class GDConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(GDConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileFaceNet(nn.Module):
    """
    MobileFaceNet adapted for Age Estimation.
    """
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, num_outputs=1, dropout_rate=0.3):
        super(MobileFaceNet, self).__init__()
        block = InvertedResidual
        input_channel = 64
        last_channel = 512

        if inverted_residual_setting is None:
            # Configuration originale
            inverted_residual_setting = [
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1],
            ]

        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        
        self.conv1 = ConvBNReLU(1, input_channel, stride=1) 
        
        self.dw_conv = DepthwiseSeparableConv(in_planes=64, out_planes=64, kernel_size=3, padding=1)
        
        features = list()
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1

                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        self.conv2 = ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        
        self.gdconv_kernel_size = 6 
        self.gdconv = GDConv(in_planes=512, out_planes=512, kernel_size=self.gdconv_kernel_size, padding=0)
        
        self.conv3 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        
        self.features = nn.Sequential(*features)

        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_outputs)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.features(x)
        x = self.conv2(x)
        
        h, w = x.shape[2], x.shape[3]
        if h != self.gdconv_kernel_size:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        else:
            x = self.gdconv(x)
            
        x = self.conv3(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1) # Flatten (Batch, 128)
        
        return self.regression_head(x)


if __name__ == "__main__":
    # Test with a dummy input
    model = MobileFaceNet()
    dummy_input = torch.randn(4, 1, 48, 48) # Batch 0f size 4, 3 channes, 48x48
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")