import torch
class ComponentwiseBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features=num_features)
    def forward(self, x: torch.Tensor):
        # BatchNorm1d expects feature dimension to be directly after Batch, i.e. [Batch x Feature x anything]
        # therefore transpose first to meet this requirement
        x = x.transpose(1,2)
        # apply batchnorm
        x = self.bn(x)
        # undo transpose
        x = x.transpose(1,2)
        return x
