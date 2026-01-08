import torch

class AlignUniformLoss(torch.nn.Module):
    def __init__(self, lambda_u=1.0):
        super().__init__()
        self.lambda_u = lambda_u

    def forward(self, z, labels):
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        align = (torch.cdist(z, z, p=2)[mask]).pow(2).mean()
        uniform = torch.log(torch.exp(-2 * torch.cdist(z, z, p=2).pow(2)).mean())

        return align + self.lambda_u * uniform
