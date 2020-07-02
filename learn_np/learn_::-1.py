import numpy as np
import cv2
#np.savetxt(filename, np.array, fmt='%g')
a=np.array([[1,2],[4,3]],dtype=np.int8)
a=[0,1,2,3,4,5,6,7]
b=a[::2]
print(b,'\n',a[::-1],'\n',a[::-2],'\n',a[5::-1])

a=np.ones((400,400,3),dtype=np.int8)*114
cv2.imshow("a.jpg",a)
cv2.waitKey(0)