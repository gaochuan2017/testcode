import torch

a=torch.randn(2,2)
print(a)
print(a.clamp(min=0,max=0.1))