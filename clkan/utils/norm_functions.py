import torch
from icecream import ic
from functorch import vmap
from enum import Enum

class Norms(Enum):
    """Enum for Normalization Types"""
    LayerNorm = "layernorm"
    BatchNorm = "batchnorm"  # BN_{\mathbb{C}}
    BatchNormNaiv = "batchnormnaiv"  # BN_{\mathbb{R}^2}
    BatchNormVar = "batchnormvar"  # BN_{\mathbb{V}} using variance
    BatchNormComponentWise = "batchnorm_comp-wise"  # apply normal BN for each node and each dimension
    BatchNormNodewise = "batchnorm_node-wise"  # apply normal BN for each node
    BatchNormDimensionwise = "batchnorm_dim-wise"  # apply normal BN for each dimension
    NoNorm = "nonorm"
# normalize each clifford dimension for each node independently
class ComponentwiseBatchNorm1d(torch.nn.Module):
    def __init__(self, num_dimensions, input_length):
        super().__init__()
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=input_length) for _ in range(num_dimensions)])
        self.num_dimensions = num_dimensions
    def forward(self, x: torch.Tensor):
        # x is either [B, I, num_features] for Clifford
        # or [B, I] with dtype complex
        # BatchNorm1d expects feature dimension to be directly after Batch, i.e. [Batch x Feature x anything]
        # therefore transpose first to meet this requirement
        if not x.is_complex():
            x = torch.stack([self.bns[i](x[:,:,i]) for i in range(self.num_dimensions)], dim=-1)
        else:
            x = torch.stack([self.bns[0](x.real), self.bns[1](x.imag)], dim=1)
            # undo split in real / imag
            x = torch.complex(x[:,0,:], x[:,1,:])
        return x
# normalize clifford dimensions independently over Nodes
class DimensionwiseBatchNorm1d(torch.nn.Module):
    def __init__(self, num_dimensions, input_length):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features=num_dimensions)
        self.num_dimensions = num_dimensions
    def forward(self, x: torch.Tensor):
        # x is either [B, I, num_features] for Clifford
        # or [B, I] with dtype complex
        # BatchNorm1d expects feature dimension to be directly after Batch, i.e. [Batch x Feature x anything]
        # therefore transpose first to meet this requirement
        if not x.is_complex():
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)
        else:
            x = torch.stack([x.real, x.imag], dim=1)
            x = self.bn(x)
            x = torch.complex(x[:,0,:], x[:,1,:])
        return x
# normalize each node in output layer independently over clifford-dimensions
class NodewiseBatchNorm1d(torch.nn.Module):
    def __init__(self, num_dimensions, input_length):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features=input_length)
        self.num_dimensions = num_dimensions
    def forward(self, x: torch.Tensor):
        # x is either [B, I, num_features] for Clifford
        # or [B, I] with dtype complex
        # BatchNorm1d expects feature dimension to be directly after Batch, i.e. [Batch x Feature x anything]
        # therefore transpose first to meet this requirement
        if not x.is_complex():
            x = self.bn(x)
        else:
            x = torch.stack([x.real, x.imag], dim=2)
            x = self.bn(x)
            x = torch.complex(x[:,:,0], x[:,:,1])
        return x

