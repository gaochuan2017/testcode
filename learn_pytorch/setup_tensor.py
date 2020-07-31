import numpy as np
import torch
import math
'''
a=np.random.randint(0,100,(2,3,2),dtype=np.int8)
print(a)
ap=torch.from_numpy(a)
'''
'''
print("ap = ",ap)
print("ap[:,None] = ",ap[None])
print("ap[:,None].shape = ",ap[None].shape)
'''
a = torch.Tensor([[0,1],[2,3],[4,5]])
b = torch.Tensor([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.4,0.4]])
#print(torch.atan(b))
'''
a = a.view(1,3,1,1,1)
b = torch.zeros(size = (1,3,4,4,1))
'''
a = a.view(3,1,2)
b = b.view(4,2)

print(a)
print(b)
print(torch.add(a,b)[...,0],'\n',torch.add(a,b)[...,1])
print(torch.add(a,b).prod(2))

scores = array([    0.94492,     0.99925,     0.70609,     0.47855,     0.46712,     0.85013,     0.96069,     0.78363,     0.97108,     0.96839,     0.94163,     0.99181,     0.88132,     0.96479,     0.95743,      0.9998,      0.9889,     0.99397,     0.97943,     0.91226,     0.78321,     0.90932,     0.49778,     0.95031,
           0.95999,     0.74344,     0.59358,     0.94379,      0.9999,      0.9953,     0.99578,     0.95698,     0.75524,     0.82947,      0.4805,     0.99289,     0.97678,     0.41514,     0.97388,     0.93086,     0.81611], dtype=float32)

a = [tensor([[2.84368e+02...='cuda:0'), tensor([[8.81891e+02...='cuda:0'), tensor([[ 4.72324e+0...='cuda:0'), tensor([[ 6.76401e+0...='cuda:0'), tensor([[ 7.84771e+0...='cuda:0'), tensor([[ 9.95665e+0...='cuda:0'), tensor([[ 3.36739e+0...='cuda:0'), tensor([[ 6.07237e+0...='cuda:0'), tensor([[3.36815e+02...='cuda:0'), tensor([[465.42377, ...='cuda:0'), tensor([[6.07578e+02...='cuda:0'), tensor([[465.17233, ...='cuda:0'), tensor([[2.80400e+02...='cuda:0'), tensor([[3.80087e+02...='cuda:0'), ...]

'''
print(ap[:,:,0].shape)
print(ap[:,:,0])
print(ap[...,0].shape)
print(ap[...,0])
'''

'''
# contiguous of np.array,please refers to this article:
# https://zhuanlan.zhihu.com/p/59767914
b=np.arange(6).reshape(2,3)
print(b.flags['C_CONTIGUOUS'])

c = torch.Tensor((3,4))
print(c)
'''