import numpy as np
import torch

def iou(box,otherboxes,isMin=False):
    box_area = (box[2]-box[0])*(box[3]-box[1])
    otherboxes_area = (otherboxes[:,2] - otherboxes[:,0]) * (otherboxes[:,3] - otherboxes[:,1])

    xx1 = torch.maximum(box[0],otherboxes[:,0])
    yy1 = torch.maximum(box[1],otherboxes[:,1])
    xx2 = torch.minimum(box[2],otherboxes[:,2])
    yy2 = torch.minimum(box[3],otherboxes[:,3])

    w,h = torch.maximum(torch.Tensor([0]),xx2-xx1),torch.maximum(torch.Tensor([0]),yy2-yy1)

    ovr_area = w*h

    if isMin:
        return ovr_area/torch.minimum(box_area,otherboxes_area)

    else:
        return ovr_area/(box_area+otherboxes_area-ovr_area)

def nms(boxes,thresh=0.5,isMin=False):

    new_boxes = boxes[boxes[:,0].argsort(descending=True)]
    #print(new_boxes)
    keep_boxes = []
    while len(new_boxes)>0:
        _box = new_boxes[0]
        keep_boxes.append(_box)
        if len(new_boxes) > 1:
            _otherboxes = new_boxes[1:]
            new_boxes = _otherboxes[torch.where(iou(_box[1:],_otherboxes[:,1:],isMin) < thresh)]
        else:
            break

    return torch.stack(keep_boxes)


if __name__ == '__main__':
    # box = torch.Tensor([0,0,4,4])
    # otherboxes = torch.Tensor([[4,4,5,5],[1,1,5,5]])
    # print(iou(box,otherboxes))
    boxes = torch.tensor([[0.5,1,1,10,10,0],[0.9,1,1,11,11,0],[0.4,8,8,12,12,0]])
    print(nms(boxes))