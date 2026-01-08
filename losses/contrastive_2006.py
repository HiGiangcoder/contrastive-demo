import torch
import torch.nn.functional as F

class ContrastiveLoss2006(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z, labels):
        dist = torch.cdist(z, z, p=2)
        label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)

        pos = label_eq.float() * dist.pow(2)
        neg = (~label_eq).float() * F.relu(self.margin - dist).pow(2)

        return (pos + neg).mean()
