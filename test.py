import numpy as np
import cv2
'''
x,y,z=np.array([12,34],dtype="int32")
print("x=%d,y=%d,z=%d" %(x,y,z))
'''

'''
a=np.full((5,6,3),1,dtype="int32")
print("value of a\n",a)
b=np.full((2,2,3),0,dtype="int32")
a[3:5,2:4]=b[0:2,0:2]
print("after set value\n",a)

img=np.random.randint(0,100,(400,400,3))
cv2.imshow("a",img)
cv2.imwrite("a.jpg",img)
'''

y=lambda x:2*x
print(y(44))
d=dict()
d[0]='a'
d[1]='b'
d[2]='c'
m=map(lambda x:2*x,'c')
print(m)