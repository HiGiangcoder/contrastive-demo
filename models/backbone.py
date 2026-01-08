import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, weight_path=None):
        super().__init__()
        net = resnet18(weights=None)

        if pretrained:
            assert weight_path is not None, "weight_path must be provided"
            state_dict = torch.load(weight_path, map_location="cpu")
            net.load_state_dict(state_dict)

        self.encoder = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class ResNet34(nn.Module):
    def __init__(self, pretrained=False, weight_path=None):
        super().__init__()
        net = resnet34(weights=None)

        if pretrained:
            assert weight_path is not None, "weight_path must be provided"
            state_dict = torch.load(weight_path, map_location="cpu")
            net.load_state_dict(state_dict)

        self.encoder = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
