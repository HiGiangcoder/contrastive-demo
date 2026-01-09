import torch
from torchvision.models import resnet18, resnet34

# ResNet18
model18 = resnet18(weights="IMAGENET1K_V1")
torch.save(model18.state_dict(), "resnet18.pth")

# ResNet34
model34 = resnet34(weights="IMAGENET1K_V1")
torch.save(model34.state_dict(), "resnet34.pth")

print("Pretrained weights saved.")
