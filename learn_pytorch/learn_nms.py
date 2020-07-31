import numpy as np
#from cv2_rectangleInsersection import *
import cv2
import torch
nms_thresh = 0.3

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

def py_cpu_nms_rbbox(dets, thresh):
    thresh = nms_thresh
    scores = dets[:, 5].numpy()
    polys = []
    areas = []
    for i in range(dets.shape[0]):
#        tm_polygon = np.array(  [dets[i][0], dets[i][1],
#                                dets[i][2], dets[i][3],
#                                dets[i][4], dets[i][5],
#                                dets[i][6], dets[i][7]],dtype = np.float)
        tm_polygon = dets[i][0:5].view(1,-1)
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = rt_bbox_iou(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def rbbox_nms(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=False, classes=None, agnostic=False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = False  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # x y w h theta conf 15classes
        # 0 1 2 3   4     5    6-20
        x = x[x[:, 5] > conf_thres]  # confidence
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 6:] *= x[..., 5:6]  # conf = obj_conf * cls_conf

        # Detections matrix nx6 (x, y, w, h ,theta, conf, cls)
        #                        0  1  2  3    4      5    6
        if multi_label:
            i, j = (x[:, 6:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = x[:, 6:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(x).all():
            x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        #c = x[:, 6] * 0 if agnostic else x[:, 6]  # classes
        #boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        #i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        i = py_cpu_nms_rbbox(dets = x[:,:6], thresh = iou_thres)

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


if __name__ == "__main__":
    a=torch.FloatTensor([400,300,300,100,0,0.8]).view(1,-1)
    b=torch.FloatTensor([200,300,400,400,0,0.9]).view(1,-1)
    c=torch.FloatTensor([300,300,200,100,0,0.6]).view(1,-1)
    d=torch.FloatTensor([300,300,200,100,90,0.7]).view(1,-1)
    dets = torch.cat((a,b,c,d),dim = 0)
    print(dets.shape)
    print(py_cpu_nms_rbbox(dets,thresh = nms_thresh))
