import torch
import torch.nn as nn
from torchvision.models import resnet18

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Giả sử input là STL-10 (96x96) hoặc CIFAR (32x32)
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Thêm MaxPool: 96->48 (STL10) hoặc 32->16 (CIFAR)

            # Block 2
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # Chuyển stride về 1 vì đã có MaxPool
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Thêm MaxPool: 48->24 (STL10) hoặc 16->8 (CIFAR)

            # Block 3
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Thêm MaxPool: 24->12 (STL10) hoặc 8->4 (CIFAR)
            
            # Thêm Block 4 (Optional - nếu dùng STL10 nên thêm để giảm sâu hơn)
            # nn.Conv2d(256, 256, 3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2), # 12->6

            nn.AdaptiveAvgPool2d(1) # Ép về 1x1 bất kể size
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# ResNet18 giữ nguyên là ổn, nó đã downsample 32 lần (96 -> 3).
class ResNet18(nn.Module):
    def __init__(self, pretrained=False, weight_path=None):
        super().__init__()
        # Lưu ý: weights=None nhanh hơn nếu không cần pretrain từ ImageNet
        net = resnet18(weights=None) 

        if pretrained and weight_path:
            state_dict = torch.load(weight_path, map_location="cpu")
            net.load_state_dict(state_dict)

        # Lấy toàn bộ trừ lớp FC cuối
        self.encoder = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)