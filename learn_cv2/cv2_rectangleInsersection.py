import cv2
import torch

def rt_bbox_iou(boxes1,boxes2):
    # input format is xywha
    # Returns the IoU of box1 to box2. box1 is nx5, box2 is nx5,return is 1xn
    ious_total=torch.zeros(boxes1.shape[0])
    assert boxes1.shape[0] == boxes2.shape[0] #'ccc'
    for num in range(0,boxes2.shape[0]):
        area1 = boxes1[num,2] * boxes1[num,3]
        area2 = boxes2[num,2] * boxes2[num,3]
        r1 = ((boxes1[num,0], boxes1[num,1]), (boxes1[num,2], boxes1[num,3]), boxes1[num,4])
        r2 = ((boxes2[num,0], boxes2[num,1]), (boxes2[num,2], boxes2[num,3]), boxes2[num,4])
        int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
        if int_pts is not None:
            order_pts = cv2.convexHull(int_pts, returnPoints=True)
            int_area = cv2.contourArea(order_pts)
            # 计算出iou
            ious = int_area * 1.0 / (area1 + area2 - int_area)
            ious_total[num] = ious
        else:
            continue
    #ious=total.cuda()
    return ious_total

#def transer_rect():

if __name__ == "__main__":
    a=torch.FloatTensor([[300,300,200,100,-30],[300,300,200,100,-30]])
    b=torch.FloatTensor([[300,300,100,200,240],[300,300,200,100,-30]])
    
    print(rt_bbox_iou(a,b))