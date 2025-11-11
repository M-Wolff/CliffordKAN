import torch
from torch_ga.clifford import GeometricAlgebra, CliffordAlgebra
import torch_ga

from cvkan import CliffordKAN

metric = [-1.0]
ga = GeometricAlgebra(metric=metric)
ca = CliffordAlgebra(metric=metric)

a = torch.tensor([[1,-1.5],[2,-1.]])
b = torch.tensor([[2,0],[-1,1.]])
#a = torch.tensor([1.0,-1.5])
#b = torch.tensor([2.0,0.0])
print(ca.num_bases)
print(ga.num_bases)

cliffkan = CliffordKAN(metric=[-1], layers_hidden=[1,5,1], use_norm=None)
print(cliffkan(a))

