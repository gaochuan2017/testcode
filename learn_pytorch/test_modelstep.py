import torch
import numpy as np
a=torch.randn(2,2)
#print(a)
#print(a.clamp(min=0,max=0.1))


xi = [0,500]
for ni in range(300):
    accumulate = max(1, np.interp(ni, xi, [1, 64]).round())
    print("ni = %d, accumulate = %d"%(ni,accumulate))