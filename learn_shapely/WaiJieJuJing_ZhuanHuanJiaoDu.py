import shapely.geometry as shg
import numpy as np
import cv2
import os
import math
img_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/images"
label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/labels"
new_label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/labels_xywha"

def draw_poly_in_picture(image,pts=None,rect=None,color=(250,0,225)):  
    if(pts is not None):
        cv2.polylines(image,[pts],True,color,1)
        #cv2.imshow("pts.jpg",image)
        #cv2.waitKey(0)
    if(rect is not None):
        box=cv2.boxPoints(rect)
        #print("rect turns to xyxy form is",box)
        box=np.int32(box)
        #print("box turns to int32",box)
        box=cv2.convexHull(box)
        #print("box sort clockwisely",box)
        cv2.polylines(image,[box],True,color,1)


def xyxy2rect(l1):#transfer a line of xyxy to rect
    pts=np.array(l1,dtype='float32').astype(np.int32)
    #ptss=[[pts[2*i],pts[2*i+1]] for i in range(4)]
    pts=pts.reshape(-1,2)
    #ptss=np.array(ptss,dtype='int32')  
    #ptss.reshape(-1,1,2)
    #print(pts==ptss)
#draw poly in the picture
    #draw_poly_in_picture(image,pts)
    rect=cv2.minAreaRect(pts)
    #print(rect)
    return rect

def write_rect(filename,rect,label):
    if(not os.path.exists(new_label_file_dir)):
        os.mkdir(new_label_file_dir)
    x=int(rect[0][0])
    y=int(rect[0][1])
    w=int(rect[1][0])
    h=int(rect[1][1])      
    theta=rect[2]/180*math.pi
    if(w<h):
        theta=theta+math.pi/2
    elif(theta<-1*math.pi/4):
        theta+=math.pi           
    with open(filename,'a') as f:
        f.write("%s %d %d %d %d %g\n"%(label,x,y,w,h,theta))

if __name__ == "__main__":
    labeltxts=[]
    images=[]
    for (d1,d2,d3) in os.walk(img_file_dir):
        print("images path is, %s, %d images found"%(d1,len(d3)))
        images=d3
    for (d1,d2,d3) in os.walk(label_file_dir):
        print("label.txt path is, %s, %d txt found"%(d1,len(d3)))
        labeltxts=d3

    for label_file in labeltxts:
        name,ext=os.path.splitext(label_file)
        label_file=label_file_dir+os.sep+label_file
        img_file=img_file_dir+os.sep+name+".png"
        if(not os.path.exists(img_file)):
            print("error: cannot find img file, ",img_file)
        image=cv2.imread(img_file)
        with open(label_file,'r') as f:
            filename=new_label_file_dir+os.sep+os.path.basename(label_file)
            lines=f.readlines()
            for l in lines:
                l=l.strip().split(' ')
                assert len(l)==10\
                #filter difficult target,which is not a full rect.
                if(int(l[-1])>0):
                    color=(0,255,255)
                else: 
                    color=(250,0,0)
                label_cur=l[-2]
                l=l[0:-2]
                #turn xyxy to xywha
                rect=xyxy2rect(l)
                #draw rect into picture
                draw_poly_in_picture(image=image,rect=rect,color=color)
                #write result into .txt
                write_rect(filename=filename,rect=rect,label=label_cur)
            #show the rect in the picture
            cv2.imshow("pts.jpg",image)
            cv2.waitKey(0)



