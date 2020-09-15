import shapely.geometry as shg
import numpy as np
import cv2
import os
import math
import random
image_size = 1024
#img_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/images"
img_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/images"
#label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/labels_origin"
label_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/labelTxt"
#new_label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/labels"
new_label_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/labels"
#img_file_dir = "/home/gaochuan/111/img"
#label_file_dir = "/home/gaochuan/111/labtxt"
#new_label_file_dir = "/home/gaochuan/111/newlabtxt"
# new_img_dir stores image that has been plotted bbox
new_img_dir = "/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/plot_labels"
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
def draw_poly_in_picture(image,pts=None,rect=None,color=(250,0,225),thickness = 1):  
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
        cv2.polylines(image,[box],True,color,thickness)



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

def plot_rotatable_rect(rect, img, color=None, showText=None, line_thickness=None):
    # Plots rbbox
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    #c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    c1,c2,theta = (int(rect[0][0]),int(rect[0][1])) , (int(rect[1][0]),int(rect[1][1])) , rect[2]
    #cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    draw_poly_in_picture(image=img,rect=(c1,c2,theta),color=color,thickness=tl)
    #draw_poly_in_picture(image=img,rect=((400,400),(300,100),45),color=color,thickness=tl)
    if showText:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(showText, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2,color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, showText, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

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
        temp = w
        w = h
        h = temp
    elif(theta<-1*math.pi/4):
        theta+=math.pi           
    with open(filename,'a') as f:
        f.write("%g %g %g %g %g %g\n"%(label_names.index(label),x,y,w,h,theta))

def generate_xywha(img_file_dir,label_file_dir,new_label_file_dir,image_size,new_img_dir):
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
                # draw orginal xyxy in img with color red
                draw_poly_in_picture(image=image,
                                pts=np.array(l,dtype='float32').astype(np.int32).reshape(-1,2),
                                color=(0,0,255),thickness=1.5)
                #draw rect into picture with blue or yellow(difficult>1)
                #draw_poly_in_picture(image=image,rect=rect,color=color,thickness=1)
                plot_rotatable_rect(img=image,rect=rect,showText=label_cur,color=color)
                #write result into .txt
                cv2.imshow("pts.jpg",image)
                #flag = cv2.waitKey(0)
                flag = ord('a')
                if(flag == ord('d')):
                    continue
                else:
                    #show the rect in the picture
                    write_rect(filename=filename,
                                rect=rect,
                                label=label_cur,
                                label_names=label_names,
                                img_size=image_size,
                                new_label_file_dir=new_label_file_dir)
            #show the rect in the picture
            #a = new_img_dir+os.sep+name+'.jpg'
            #print(a)
            #cv2.imwrite(a,image)


if __name__ == "__main__":
    generate_xywha(img_file_dir      = img_file_dir,
                   label_file_dir    = label_file_dir,
                   new_label_file_dir= new_label_file_dir,
                   new_img_dir       = new_img_dir,
                   image_size        = image_size)





