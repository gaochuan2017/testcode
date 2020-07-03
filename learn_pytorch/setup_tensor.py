import numpy as np
import torch

a=np.random.randint(0,100,(3,4),dtype=np.int8)
print(a)
ap=torch.from_numpy(a)
print(ap)

# contiguous of np.array,please refers to this article:
# https://zhuanlan.zhihu.com/p/59767914
b=np.arange(6).reshape(2,3)
print(b.flags['C_CONTIGUOUS'])