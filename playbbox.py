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

def play_bbox(src,label,save):
    image=cv2.imread(src)
    with open(label) as f:
        lines=f.readlines()
#        list(map(print,lines))
        for l in lines:
            l=l.strip().split(' ')
            print(l)
            if(len(l)==10):
                pts=np.array([[l[0],l[1]],
                              [l[2],l[3]],
                              [l[4],l[5]],
                              [l[6],l[7]]],dtype='int32')
                cv2.polylines(image,[pts],1,(0,0,255),2)
#        cv2.imshow("a",image)
#        cv2.waitKey(0)
        cv2.imwrite("result.jpg",image)

if __name__ == "__main__":
    play_bbox(src="image/P2598.png",label="image/P2598.txt",save=True)

