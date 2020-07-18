import shapely.geometry as shg
import numpy as np
import cv2
import os
import math
image_size = 512
img_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/images"
#label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/labels_origin"
label_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/labelTxt"
#new_label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/labels"
new_label_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/labels"
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

def write_rect(filename,rect,label,label_names,img_size,new_label_file_dir):
    if(not os.path.exists(new_label_file_dir)):
        os.mkdir(new_label_file_dir)
    x=float(rect[0][0]/img_size)
    y=float(rect[0][1]/img_size)
    w=float(rect[1][0]/img_size)
    h=float(rect[1][1]/img_size)      
    theta=rect[2]/180*math.pi
    if(w<h):
        theta=theta+math.pi/2
    elif(theta<-1*math.pi/4):
        theta+=math.pi           
    with open(filename,'a') as f:
        f.write("%g %g %g %g %g %g\n"%(label_names.index(label),x,y,w,h,theta))

def generate_xywha(img_file_dir,label_file_dir,new_label_file_dir,image_size):
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
                # draw orginal xyxy in img with color red
                draw_poly_in_picture(image=image,
                                pts=np.array(l,dtype='float32').astype(np.int32).reshape(-1,2),
                                color=(0,0,255))
                #draw rect into picture with blue or yellow(difficult>1)
                draw_poly_in_picture(image=image,rect=rect,color=color)
                #write result into .txt
                '''write_rect(filename=filename,
                            rect=rect,
                            label=label_cur,
                            label_names=label_names,
                            img_size=image_size,
                            new_label_file_dir=new_label_file_dir)'''
            #show the rect in the picture
            cv2.imshow("pts.jpg",image)
            cv2.waitKey(0)

if __name__ == "__main__":
    generate_xywha(img_file_dir      = img_file_dir,
                   label_file_dir    = label_file_dir,
                   new_label_file_dir= new_label_file_dir,
                   image_size        = image_size)





