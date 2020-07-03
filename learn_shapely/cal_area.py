import shapely.geometry as shg
import numpy as np
import cv2
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
                              [l[6],l[7]]],dtype='float32').astype(np.int32)
                cv2.polylines(image,[pts],1,(0,0,255),2)
        cv2.imshow("b.jpg",image)
        cv2.waitKey(0)
        if save==True:
            cv2.imwrite("result.jpg",image)

label_file="./image/P0706__1__0___0.txt"
img_file="./image/P0706__1__0___0.png"
with open(label_file,'r') as f:
    lines=f.readlines()
    '''
    for l in lines:
        l=l.strip().split(' ')
        l=l[0:-2]
        l1=[float(x) for x in l]
        pts=np.array(l1,dtype='float32').astype(np.int)
        ptss=[(pts[2*i],pts[2*i+1]) for i in range(4)]
        #print(ptss)
    '''
    l=lines[-1].strip().split(' ')
    l=l[0:-2]
    l1=[float(x) for x in l]
    pts=np.array(l1,dtype='float32').astype(np.int32)
    ptss=[(pts[2*i],pts[2*i+1]) for i in range(4)]
    print(ptss)
    print(shg.Polygon(ptss).area)
    image=cv2.imread(img_file)
    cv2.polylines(image,[np.array(ptss)],1,(0,0,255),2)
    cv2.imshow("b.jpg",image)
    cv2.waitKey(0)

