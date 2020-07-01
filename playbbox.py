import numpy as np
import shapely.geometry as shg
import cv2
def test_play():
    pts=([0,0],[100,0],[100,100],[0,100])
    img=cv2.imread("image/P2598.png")
    img=np.zeros([400,400,3],np.uint8)
    img[:,:,0]=np.full([400,400],255,dtype=np.uint8)
    cv2.imshow("aa",img)
    cv2.waitKey(0)

    cv2.polylines(img,[np.array(pts,dtype='int32')],1,(255,0,255),3)
    cv2.imshow("test.jpg",img)
    cv2.waitKey(0)
    img=cv2.imwrite("test.jpg",img)
    pass

def play_bbox():

    return

if __name__ == "__main__":
    
    return