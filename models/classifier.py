import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, backbone, feat_dim, emb_dim, num_classes):
        super().__init__()
        self.encoder = backbone
        self.projector = nn.Linear(feat_dim, emb_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        z = F.normalize(self.projector(feat), dim=1)
        logits = self.classifier(feat)
        return logits, z
