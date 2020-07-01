import numpy as np
import cv2
#funtion order_points only fits the situation that target is rect without rotation 
def order_points(pts):
    rect=np.zeros((4,2),dtype="int32")
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]
    a=np.diff(pts,axis=1)
    rect[3]=pts[np.argmax(a)]
    rect[1]=pts[np.argmin(a)]
    return rect

#def convex2xywha(boxes1, boxes2, use_gpu=False, gpu_id=1):

if __name__ == "__main__":
    img=cv2.imread("coco.jpg")
    pts=np.random.randint(30,300,size=(4,2))
    img=np.zeros((512,512,3),np.uint8)
    image=img.copy()
    #pts=np.array([[30,70],[80,20],[10,10],[100,120]],dtype="int32")
    print(pts)
    print("show pts in the picture")
    #for i in range(pts.shape[0]):
    #    x,y=pts[i][0],pts[i][1]
    #    cv2.circle(image,(x,y),5,color=(0,233,255))
    [cv2.circle(image,(pts[i][0],pts[i][1]),5,color=(0,233,255)) for i in range(pts.shape[0])]
    #pts=order_points(pts)
    pts=cv2.convexHull(pts)
    #pts=pts.reshape(-1,1,2,)
    print(pts.shape)
    for i in range(pts.shape[0]):
        x,y=pts[i][0][0],pts[i][0][1]
        cv2.circle(image,(x,y),3,color=(0,0,255),thickness=-1)
    cv2.polylines(image,np.array([pts], np.int32),True,(250,0,255),1)
    rect=cv2.minAreaRect(pts)
    print("rect is xywha,",rect)
    box=cv2.boxPoints(rect)
    print("rect turns to xyxy form is",box)
    box=np.int32(box)
    print("box turns to int32",box)
    box=order_points(box)
    print("box sort clockwisely",box)
    cv2.polylines(image,[box],True,(200,200,200),1)
    cv2.imshow("clockwise pts",image)
    cv2.waitKey(0)

#    clock_pts=ns(pts)
#    rect=cv2.minAreaRect(clock_pts)
#    rect_pts=cv2.boxpoints()
    
