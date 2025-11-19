import torch.nn as nn
import torch

class CNNBlock(nn.Module):

    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)
        # self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)
        # self.bn2   = nn.BatchNorm2d(channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
            # nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class LinearBlock(nn.Module):

    def __init__(self, dim, mlp_hidden=128):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, mlp_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, dim, bias=False),
        )
        self.shortcut = nn.Linear(dim, dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.seq(x)
        out += identity
        out = self.relu(out)
        return out

def calc_out_shape(size, kernel, stride):
    return (size + 2 * (kernel // 2) - kernel) // stride + 1

class ResNetCNN(nn.Module):
    
    def __init__(self, num_blocks=1, num_classes=10, base_channels=3, kernel_size=9, stride=1, mlp_hidden=128):
        super().__init__()

        self.conv_in = nn.Conv2d(
            1, base_channels, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False
        )
        self.bn_in = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([
            CNNBlock(base_channels, kernel_size=kernel_size, stride=stride)
                for _ in range(num_blocks)
        ])
        size = 28
        for _ in range(num_blocks+1):
            size = calc_out_shape(size, kernel_size, stride)
        assert size > 0, f"Too many downsamples: feature map negative ({size})."
        self.flat_dim = base_channels * size * size
        self.fc = nn.Linear(self.flat_dim, num_classes)
        
    def forward(self, x):
        out = self.relu(self.bn_in(self.conv_in(x)))
        for block in self.blocks:
            out = block(out)
        out = out.view(out.size(0), -1)
        print(f'{out.shape=}')
        out = self.fc(out)
        return out

class ResNetFNN(nn.Module):
    
    def __init__(self, num_blocks=1, num_classes=10, mlp_hidden=128):
        super().__init__()
        self.linear_in = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(28*28, mlp_hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            LinearBlock(mlp_hidden, mlp_hidden=mlp_hidden)
                for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(mlp_hidden, num_classes)
        
    def forward(self, x):
        out = self.linear_in(x)
        for block in self.blocks:
            out = block(out)
        print(f'{out.shape=}')
        out = self.fc(out)
        return out


if __name__ == "__main__":
    # CNN
    n = 1
    model_cnn = ResNetCNN(num_blocks=n, base_channels=8, kernel_size=7, stride=3)
    x = torch.randn(41, 1, 28, 28)
    print("CNN output:", model_cnn(x).shape)

    # Linear
    model_lin = ResNetFNN(num_blocks=n, mlp_hidden=128)
    x = torch.randn(31, 1, 28, 28)
    print("Linear output:", model_lin(x).shape)