from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from yolo.utils import make_416_image
import numpy as np
from yolo.config import *

tf=transforms.Compose([
    transforms.ToTensor()
])

def one_hot(cls_num,i):
    rst = np.zeros(cls_num)
    rst[i] = 1
    return rst

class YoloDataSet(Dataset):
    def __init__(self):
        with open('data/datas.txt','r') as f:
            self.datas = f.readlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = self.datas[item]
        data_list = data[:-1].split(" ")

        _boxes = np.array([float(x) for x in data_list[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)

        img = make_416_image(os.path.join('data/imgs',data_list[0]))
        w,h = img.size
        case = 416/w
        img = img.resize((DATA_WIDTH,DATA_HEIGHT))
        img_data = tf(img)
        labels = {}
        for feature_size,_antors in antors.items():
            labels[feature_size] = np.zeros((feature_size,feature_size,3,5+CLASS_NUM))
            #print(labels[feature_size].shape)
            for box in boxes:
                cls,cx,cy,w,h = box
                cx,cy,cw,ch = cx*case,cy*case,w*case,h*case
                #print(cls,cx,cy,w,h)
                _x,x_index = math.modf(cx*feature_size/DATA_WIDTH)
                _y,y_index = math.modf(cy*feature_size/DATA_HEIGHT)
                for i,antor in enumerate(_antors):
                    area = w*h
                    iou = min(area,ANTORS_AREA[feature_size][i])/max(area,ANTORS_AREA[feature_size][i])
                    p_w,p_h = cw/antor[0],ch/antor[1]

                    labels[feature_size][int(y_index),int(x_index)] = np.array([iou,_x,_y,np.log(p_w),np.log(p_h),*one_hot(CLASS_NUM,int(cls))])

        return labels[13],labels[26],labels[52],img_data




if __name__ == '__main__':
    dataset = YoloDataSet()
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    print(dataset[0][3].shape)