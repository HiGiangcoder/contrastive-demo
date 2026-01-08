import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, z, labels):
        sim = torch.matmul(z, z.T) / self.tau
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        sim_exp = torch.exp(sim)
        pos = sim_exp * mask
        neg = sim_exp * (~mask)

        return -torch.log(pos.sum(1) / (pos.sum(1) + neg.sum(1))).mean()
