import torch
import torch.nn.functional as F

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z, labels):
        dist = torch.cdist(z, z, p=2)

        loss = 0.0
        count = 0

        for i in range(z.size(0)):
            pos = dist[i][labels == labels[i]]
            neg = dist[i][labels != labels[i]]

            if len(pos) > 1 and len(neg) > 0:
                loss += F.relu(pos.mean() - neg.mean() + self.margin)
                count += 1

        return loss / max(count, 1)
