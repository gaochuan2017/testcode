import torch
import numpy as np
import math
a=np.array([1,3,1,2,1,5],dtype=np.int)
'''
a1=a.reshape(-1,1,2)
a1=torch.from_numpy(a1)
#print("a1 is : \n",a1)
b=np.array([1,1,2,2,3,3,4,4],dtype=np.int)
b1=b.reshape(1,-1,2)
b1=torch.from_numpy(b1)
#print("b1 is : \n",b1)

#print(torch.min(a1,b1))
print(a1)
print(a1.prod(0))
print(a1.prod(1))
'''

print(math.atan([1,0,-1]))