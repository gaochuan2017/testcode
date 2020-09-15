import numpy as np
import cv2
pts=([0,0],[100,0],[100,100],[0,100])
#img=cv2.imread("image/P2598.png")
img=np.zeros([400,600,3],np.uint8)
img[:,:,0]=np.full([400,600],255,dtype=np.uint8)
cv2.imshow("aa",img)
a = cv2.waitKey(0)
#print(a)
if(a == ord('d')): print("input d")
elif (a == ord('c')):
    print("input c")
else: print(a)