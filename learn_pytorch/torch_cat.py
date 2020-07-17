import torch
import numpy as np

a=torch.ones(6,2,dtype=torch.float32)
print(a.type)
b=torch.randn(6,2)
c=torch.zeros(6,1)
c1=torch.randn(6,)
print(c1)
print(a.type == b.type)
c1=c1.view(-1,1)
print(c1)
d=torch.cat((a,c1),dim=1)
print(d)
