import numpy as np
import torch
'''
a = np.random.randint(low =1,high = 100,size =(5,21))
a = torch.from_numpy(a)
print(a)
i,j = a.max(1)
print(i,i.unsqueeze(1))
print(torch.isfinite(j).all(1))
print(i.all())
'''
a = torch.zeros((3,2,4))
b = torch.Tensor([1,2,3]).view(3,1,1)
print(a)
print(b)
print(torch.add(a,b))
'''
a = np.random.random([4,5])
a = torch.from_numpy(a)
b = torch.exp(a)
print(torch.Tensor([1,3]))
print(a)
print(torch.log(b))
'''
'''
print(a.argsort())
print(a.argsort()[::-1])
b = [a[a.argsort()[::-1][i]] for i in range(5)]
print("b = ",b,'\n')
b = np.array(b)

ind = np.where(b >= 5)[0]
ind1= np.where(b >= 5)
print(b,ind)
'''



