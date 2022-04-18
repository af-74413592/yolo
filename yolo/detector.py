import torch
from torch import nn
from yolo.yolo_v3_net import Yolo_V3_Net
from PIL import Image,ImageDraw
from yolo.utils import make_416_image
from yolo.dataset import tf
from yolo.config import *
import json
from yolo.nms import nms

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = 'params/net.pt'
        self.net = Yolo_V3_Net().to(self.device)
        self.net.load_state_dict(torch.load(self.weights))
        self.net.eval()

    def forward(self,input,thresh,anchors,case):
        output_13,output_26,output_52 = self.net(input)
        index_13,bias_13 = self.get_index_and_bias(output_13,thresh)
        boxes_13 = self.get_true_position(index_13,bias_13,32,anchors[13],case)
        index_26, bias_26 = self.get_index_and_bias(output_26, thresh)
        boxes_26 = self.get_true_position(index_26, bias_26, 16, anchors[26],case)
        index_52, bias_52 = self.get_index_and_bias(output_52, thresh)
        boxes_52 = self.get_true_position(index_52, bias_52, 8, anchors[52],case)

        return torch.cat([boxes_13,boxes_26,boxes_52],dim=0)


    def get_index_and_bias(self,output,threshold):
        output = output.permute(0,2,3,1)
        output = output.reshape(output.size(0),output.size(1),output.size(2),3,-1)

        mask = output[...,0] > threshold
        index = mask.nonzero() #True值索引
        bias = output[mask]

        return index,bias

    def get_true_position(self,index,bias,t,anchors,case):
        anchors = torch.Tensor(anchors)

        a = index[:,3]

        cy = (index[:,1].float()+bias[:,2].float())*t/case
        cx = (index[:,2].float()+bias[:,1].float())*t/case
        w = anchors[a,0].to(self.device)*torch.exp(bias[:,3])/case
        h = anchors[a,1].to(self.device)*torch.exp(bias[:,4])/case

        p = bias[:,0]
        cls_p = bias[:,5:]
        cls_index = torch.argmax(cls_p,dim=1)

        return torch.stack([torch.sigmoid(p),cx,cy,w,h,cls_index],dim=1)

if __name__ == '__main__':
    detector = Detector()
    img =  Image.open('data/imgs/000010.jpg')
    _img = make_416_image('data/imgs/000010.jpg')
    ori_max=max(_img.size)
    case = 416/ori_max
    _img = _img.resize((416,416))
    _img = tf(_img).unsqueeze(dim=0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    results = detector(_img,0.3,antors,case)
    draw = ImageDraw.Draw(img)
    class_count = json.load(open('data/class_count.json','r'))
    class_num = {v:k for k,v in class_count.items()}
    boxes = []
    for rst in results:

        x1,y1,x2,y2 = rst[1]-0.5*rst[3],rst[2]-0.5*rst[4],rst[1]+0.5*rst[3],rst[2]+0.5*rst[4]
        #print(x1,y1,x2,y2)
        boxes.append([rst[0],x1,y1,x2,y2,rst[5]])

    new_boxes = nms(torch.Tensor(boxes),0.2)
    for new_box in new_boxes:
        print(f'class:{class_num[int(new_box[5].item())]},p:{str(new_box[0].item())[:4]},x1:{int(new_box[1])},y1:{int(new_box[2])},x2:{int(new_box[3])},y2:{int(new_box[4])}')

        draw.text((new_box[1],new_box[2]),str(class_num[int(new_box[5].item())])+str(new_box[0].item())[:4])
        draw.rectangle((new_box[1],new_box[2],new_box[3],new_box[4]),outline='red',width=1)

    img.show()