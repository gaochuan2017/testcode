import numpy as np
#np.savetxt(filename, np.array, fmt='%g')
a=np.array([[1,2],[4,3]],dtype=np.int8)
a=[0,1,2,3,4,5,6,7]
b=a[::2]
print(b,'\n',a[::-1],'\n',a[::-2],'\n',a[5::-1])