import cv2
import numpy as np
import torch
import math
import os

image_size = 1024
#img_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/images"
img_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/images"
#label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/labels_origin"
label_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/labelTxt"
#new_label_file_dir="/home/gaochuan/object_detection/dataset/dota_8p/train/labels"
new_label_file_dir="/home/gaochuan/object_detection/dataset/dota_512/dota_512_split/train/labels"
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

def recf_from_label(line):
    return ((float(line[1])*image_size,float(line[2])*image_size), 
            (float(line[3])*image_size,float(line[4])*image_size) ,
            float(line[5])*180/math.pi)

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
        cv2.putText(img, showText, (c1[0], c1[1] - 2), 0, tl / 3, [225, 144, 30], thickness=tf, lineType=cv2.LINE_AA)

def draw_label_file(img_file_dir,label_file_dir,new_label_file_dir,image_size,new_img_dir):
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
            print("%s image file don't exist, ",img_file)
        image=cv2.imread(img_file)

        filename=new_label_file_dir+os.sep+os.path.basename(label_file)
        if(not os.path.exists(filename)):
            # remove img_file
            print("There's no any targets in picture %s "%img_file)
            continue
        new_img_file = new_img_dir+os.sep+name+'.png'
        cv2.imwrite(new_img_file,image)
        with open(filename,'r') as f:           
            lines=f.readlines()
            if(not len(lines)):
                print("empty file!")
            else:
                print("empty file!")
            for l in lines:
                l=l.strip().split(' ')
                assert len(l)== 6
                label_cur = label_names[int(l[0])]
                rect = recf_from_label(l)
                color = (0,252,124)
                #rect=xyxy2rect(l)
                plot_rotatable_rect(img=image,rect=rect,showText=label_cur,color=color)
            cv2.imshow("bboxes on %s"%name,image)
            cv2.waitKey(300)

            #print(a)
            #cv2.imwrite(a,image)

def generate_train_txt(img_file_dir,label_file_dir,train_txt_dir):
    images=[]
    for (d1,d2,d3) in os.walk(img_file_dir):
        print("images path is, %s, %d images found"%(d1,len(d3)))
        images = d3

    txt_file = train_txt_dir+'/'+'train.txt'
    if(not os.path.exists(txt_file)):
        os.system('touch %s'%txt_file)
    with open(txt_file , 'w') as f:
        for img_file in images:
            f.write(img_file_dir + os.sep + img_file+'\n')

def remove_not_1024_img(img_file_dir,label_file_dir,train_txt_dir,img_size):
    images = []
    for (d1,d2,d3) in os.walk(img_file_dir):
        print("images path is, %s, %d images found"%(d1,len(d3)))
        images = d3
    
    for img_file in images:
        img_file_path = img_file_dir + os.sep + img_file 
        img = cv2.imread(img_file_path)
        if( not(img.shape[0] == img_size and img.shape[1] == img_size)):
            print("img %s width or height not %s"%(img_file,img_size))
            label_file_path = label_file_dir + os.sep + img_file.replace('.png','.txt')
            if(os.path.exists(label_file_path)):
                os.system('rm %s'%label_file_path)
            os.system('rm %s'%img_file_path)

def remove_no_label_img(img_file_dir,label_file_dir):
    images = []
    for (d1,d2,d3) in os.walk(img_file_dir):
        print("images path is, %s, %d images found"%(d1,len(d3)))
        images = d3
    
    for img_file in images:
        img_file_path = img_file_dir + os.sep + img_file 
        label_file_path = label_file_dir + os.sep + img_file.replace('.png','.txt')
        if(not os.path.exists(label_file_path)):
            #os.system('rm %s'%img_file_path)
            print(label_file_path)

def examine_label_no(img_file_dir,label_file_dir):
    labels = []
    for (d1,d2,d3) in os.walk(label_file_dir):
        print("label path is, %s, %d labelfiles found"%(d1,len(d3)))
        labels = d3

    for label_file in labels:
        label_file_path = label_file_dir + os.sep + label_file
        img_file_path = img_file_dir + os.sep + label_file.replace('.txt','.png')
        with open(label_file_path,'r') as f:
            l = f.readlines()
            if(len(l) == 0):
                print(label_file_path)
 


def remove_label_more_than_one(img_file_dir,label_file_dir,train_txt_dir,img_size):
    #images = []
    labels = []
    for (d1,d2,d3) in os.walk(label_file_dir):
        print("label path is, %s, %d labelfiles found"%(d1,len(d3)))
        labels = d3

    for label_file in labels:
        label_file_path = label_file_dir + os.sep + label_file
        img_file_path = img_file_dir + os.sep + label_file.replace('.txt','.png')
        with open(label_file_path,'r') as f:
            l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            flags = l.shape[1] == 6 and (l[:,0:5] >= 0).all() and (l[:, 1:5] <= 1).all() \
            and (l[:,5]>=-1/4*math.pi).all() and (l[:,5]<=3/4*math.pi).all()
            if(not flags):
                print(label_file_path)
                if(os.path.exists(img_file_path)):
                    os.system('rm %s'%img_file_path)
                    os.system('rm %s'%label_file_path)

def compute_label_NO(img_file_dir,label_file_dir,train_txt_dir,img_size):
    labels = []
    label_count = np.zeros(15)
    for (d1,d2,d3) in os.walk(label_file_dir):
        print("label path is, %s, %d labelfiles found"%(d1,len(d3)))
        labels = d3
    for label_file in labels:
        label_file_path = label_file_dir + os.sep + label_file
        with open(label_file_path,'r') as f:
            l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            for i in range(l.shape[0]):
                label_count[int(l[i,0])] += 1.0
    return label_count

def extract_ori_label(img_file_dir,label_file_dir,new_label_file_dir):
    images = []
    for (d1,d2,d3) in os.walk(img_file_dir):
        print("images path is, %s, %d images found"%(d1,len(d3)))
        images = d3

    for img_file in images:    
        img_file_path = img_file_dir + os.sep + img_file 
        label_file_path = label_file_dir + os.sep + img_file.replace('.png','.txt')
        new_label_file_path = new_label_file_dir + os.sep + img_file.replace('.png','.txt')
        os.system("cp %s %s"%(label_file_path,new_label_file_path))

def remove_poly_label_less8(img_file_dir,label_file_dir,train_txt_dir = None,img_size = 1024):
    #images = []
    labels = []
    for (d1,d2,d3) in os.walk(label_file_dir):
        print("label path is, %s, %d labelfiles found"%(d1,len(d3)))
        labels = d3

    for label_file in labels:
        label_file_path = label_file_dir + os.sep + label_file
        img_file_path = img_file_dir + os.sep + label_file.replace('.txt','.png')
        with open(label_file_path,'r') as f:
            for l in f.read().splitlines():
                l = l.split(' ')
                flags = len(l) == 10
            if(not flags):
                print(label_file_path)
                '''
                if(os.path.exists(img_file_path)):
                    os.system('rm %s'%img_file_path)
                    os.system('rm %s'%label_file_path)
                '''

if __name__ == "__main__":
    '''
    draw_label_file(img_file_dir      = img_file_dir,
        label_file_dir    = label_file_dir,
        new_label_file_dir= new_label_file_dir,
        new_img_dir       = new_img_dir,
        image_size        = image_size)
    
    '''
    '''
    generate_train_txt(img_file_dir = r'/home/gaochuan/object_detection/dataset/dota_8p/train/images',
                        label_file_dir=r'/home/gaochuan/object_detection/dataset/dota_8p/train/labels',
                        train_txt_dir = r'/home/gaochuan/object_detection/dataset/dota_8p')   
    '''
    '''
    remove_not_1024_img(img_file_dir = r'/home/gaochuan/object_detection/dataset/dota_512_origin/dota_512_split/IMAGE',
                        label_file_dir=r'/home/gaochuan/object_detection/dataset/dota_512_origin/dota_512_split/LABEL',
                        train_txt_dir = r'/home/gaochuan/object_detection/dataset/dota_8p',
                        img_size = 1024)
    '''
    '''
    remove_no_label_img(img_file_dir = r'/home/gaochuan/object_detection/dataset/dota_8p/train/images',
                    label_file_dir=r'/home/gaochuan/object_detecti/home/gaochuan/object_detection/dataset/dota_8p/train/labelson/dataset/dota_8p//home/gaochuan/object_detection/dataset/dota_8p/train/labelstrain/labels',
                    train_txt_dir = r'/home/gaochuan/object_detection/dataset/dota_8p',
                    img_size = 1024)

    remove_label_more_than_one(img_file_dir = r'/home/gaochuan/object_detection/dataset/dota_8p/train/images',
                label_file_dir=r'/home/gaochuan/object_detection/dataset/dota_8p/train/labels',
                train_txt_dir = r'/home/gaochuan/object_detection/dataset/dota_8p',
                img_size = 1024)

    generate_train_txt(img_file_dir = r'/home/gaochuan/object_detection/dataset/dota_8p/train/images',
                        label_file_dir=r'/home/gaochuan/object_detection/dataset/dota_8p/train/labels',
                        train_txt_dir = r'/home/gaochuan/object_detection/dataset/dota_8p') 
    '''
    '''
    count = compute_label_NO(img_file_dir = r'/home/gaochuan/object_detection/dataset/dota_8p/train/images',
                label_file_dir=r'/home/gaochuan/object_detection/dataset/dota_8p/train/labels',
                train_txt_dir = r'/home/gaochuan/object_detection/dataset/dota_8p',
                img_size = 1024)
        
    for i in range(15):
        print("class %s has %d label targets total in dataset\n"%(label_names[i] , int(count[i])))
    '''
    '''
    extract_ori_label(img_file_dir = r"/home/gaochuan/object_detection/dataset/dota_8p/train/images",
                        label_file_dir = r"/home/gaochuan/object_detection/dataset/dota_512_origin/dota_512_split/labelTxt",
                        new_label_file_dir = r"/home/gaochuan/object_detection/dataset/dota_8p/train/ORI_label")
    '''
    '''
    examine_label_no(img_file_dir = r"/home/gaochuan/object_detection/dataset/dota_8p/train/images",
                        label_file_dir = r"/home/gaochuan/object_detection/dataset/dota_8p/train/labelTxt",
    )
    '''
    remove_poly_label_less8(img_file_dir = r"/home/gaochuan/object_detection/dataset/dota_8p/train/prepare_for_json/images",
                        label_file_dir = r"/home/gaochuan/object_detection/dataset/dota_8p/train/prepare_for_json/labelTxt",
    )