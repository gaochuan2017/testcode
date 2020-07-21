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
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def gt_anchor_iou(anchors,t,anchor_theta):
    #anchors.shape = (3,2) t.shape = (N,7) anchor_theta.shape =(3,) 
    boxes_1 = torch.full_like(anchors, 0.5)
    boxes_1 = torch.cat((boxes_1, anchors),1)
    gxy = t[:, 2:4]  # grid xy
    gwh = t[:, 4:6]  # grid wh
    gtheta = t[:,6].view(-1,1) #grid theta 'ccc'
    gij = (gxy - 0).long()
    gi, gj = gij.T  # grid xy indices
    boxes_2 = torch.cat((gxy - gij, gwh), 1)
    area1 = boxes_1[:,2] * boxes_1[:,3]
    area2 = boxes_2[:,2] * boxes_2[:,3]
    box1 = xywh2xyxy(boxes_1)
    box2 = xywh2xyxy(boxes_2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    iou = inter / (area1[:, None] + area2 - inter)
    iou2= box_iou(box1,box2)
    print(torch.abs(iou-iou2)<1E-2,'\n',iou,'\n',iou2)
    return iou

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

if __name__ == "__main__":
#    a=torch.FloatTensor([[200,100,-30],[0,0,300,300,200,100,-30]])
    a=torch.FloatTensor([[200,100],[200,100]])
    b=torch.FloatTensor([[0,0,0.5,0.5,100,200,240],[0,0,0.5,0.5,100,100,-30]])
    anchor_theta = torch.Tensor([0,0,0]).reshape(-1,1)
    c=torch.FloatTensor([[300,300,200,100,0],[300,300,200,100,0]])
    d=torch.FloatTensor([[300,300,100,200,0],[300,300,100,100,0]])
    
    a = torch.randn(3,2).clamp(min = 0.2)
    b = torch.randn(5,7)
    b[:,4:6] = b[:,4:6].clamp(min = 0.2)
    print(b)
    gt_anchor_iou(a,b,anchor_theta=anchor_theta)
    #print("box_iou = ",box_iou(a[:,2:6],b[:,2:6]))