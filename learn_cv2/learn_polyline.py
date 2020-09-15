import cv2
import numpy as np
import torch
import math
import os
'''
image_size = 1024
img_file_dir = "/home/gaochuan/111/images"
label_file_dir = "/home/gaochuan/111/labels"
new_label_file_dir = "/home/gaochuan/111/newlabtxt"
# new_img_dir stores image that has been plotted bbox
new_img_dir = "/home/gaochuan/111/newimg"
'''
'''
image_size = 512
#img_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/images"
img_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/images"
#label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/labels_origin"
label_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/labelTxt"
#new_label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/labels"
new_label_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/labels"
new_img_dir = "/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/plot_labels"
'''

image_size = 1024
img_file_dir = "/home/gaochuan/object_detection/dataset/dota_mini/dotamini_split/images"
label_file_dir = "/home/gaochuan/object_detection/dataset/dota_mini/dotamini_split/labelTxt"
new_label_file_dir = ""
new_img_dir = "/home/gaochuan/object_detection/dataset/dota_mini/dotamini_split/new_images"

label_names=["plane",
            "ship",
            "storage-tank",
            "baseball-diamond",
            "tennis-court",
            "basketball-court",
            "ground-track-field",
            "harbor",
            "bridge",
            "large-vehicle",
            "small-vehicle",
            "helicopter",
            "roundabout",
            "soccer-ball-field",
            "swimming-pool"]

def draw_poly_in_picture(image,pts=None,rect=None,color=(250,0,225),thickness=3):  
    if(pts is not None):
        cv2.polylines(image,[pts],True,color,3)
        #cv2.imshow("pts.jpg",image)
        #cv2.waitKey(0)
    if(rect is not None):
        box=cv2.boxPoints(rect)
        #print("rect turns to xyxy form is",box)
        box=np.int32(box)
        #print("box turns to int32",box)
        box=cv2.convexHull(box)
        #print("box sort clockwisely",box)
        cv2.polylines(image,[box],True,color,thickness)

def xyxy2rect(l1):#transfer a line of xyxy to rect
    pts=np.array(l1,dtype='float32').astype(np.int32)
    #ptss=[[pts[2*i],pts[2*i+1]] for i in range(4)]
    pts=pts.reshape(-1,2)
    rect=cv2.minAreaRect(pts)
    #print(rect)
    return rect

def convert_rect(rect,img_size):
    '''
    x=float(rect[0][0]/img_size)
    y=float(rect[0][1]/img_size)
    w=float(rect[1][0]/img_size)
    h=float(rect[1][1]/img_size)  
    '''
    x=float(rect[0][0])
    y=float(rect[0][1])
    w=float(rect[1][0])
    h=float(rect[1][1])     
    theta=rect[2]/180*math.pi
    if(w<h):
        theta=theta+math.pi/2
        temp = w
        w = h
        h = temp
    elif(theta<-1*math.pi/4):
        theta+=math.pi  
    return ((x,y),(w,h),theta*180/math.pi)

def bianli(img_file_dir,label_file_dir,new_label_file_dir,image_size,new_img_dir):
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
            #filename=new_label_file_dir+os.sep+os.path.basename(label_file)
            lines=f.readlines()
            for l in lines:
                l=l.strip().split(' ')
                assert len(l)==10
                #filter difficult target,which is not a full rect.
                if(int(l[-1])>0):
                    color=(0,255,255)
                else: 
                    color=(250,0,0)
                label_cur=l[-2]
                l=l[0:-2]
                #turn xyxy to xywha
                rect=xyxy2rect(l)
                draw_poly_in_picture(image=image,rect=rect,color=color,thickness=2)
                rect=convert_rect(rect,img_size=image_size)
                draw_poly_in_picture(image=image,rect=rect,color=(123,34,200),thickness=2)
            cv2.imshow("pts.jpg",image)
            cv2.waitKey(0)

if __name__ == "__main__":
    bianli(img_file_dir      = img_file_dir,
            label_file_dir    = label_file_dir,
            new_label_file_dir= new_label_file_dir,
            new_img_dir       = new_img_dir,
            image_size        = image_size)





























'''
imgW=1024
imgH=1024
img_b= np.ones(shape=(imgW,imgH))*255
#img_g= np.zeros(shape=(400,400))
#img_r= np.zeros(shape=(400,400)) 
img = np.zeros(shape = (imgW,imgH,3))
img[:,:,0] = img_b
#cv2.imwrite("a.jpg",img)


pts=([0,0],[100,0],[100,100],[0,100])
rect1 = ((500,500),(300,100),45)
rect2 = ((500,500),(300,100),-75)
rect3 = ((500,500),(300,100),230)
draw_poly_in_picture(image=img,rect=rect1)
draw_poly_in_picture(image=img,rect=rect2,color=[np.random.randint(0,255) for i in range(3)])
draw_poly_in_picture(image=img,rect=rect3,color=[np.random.randint(0,255) for i in range(3)])
cv2.imshow("a.jpg",img)
cv2.waitKey(0)
'''