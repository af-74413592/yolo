from PIL import Image
import numpy as np
import os

def make_416_image(img_path):
    img = Image.open(img_path)
    w,h = img.size[0],img.size[1]
    temp = max(w,h)
    mask = Image.new(mode='RGB',size=(temp,temp),color=(0,0,0))
    mask.paste(img,(0,0))
    return mask

def IOU(x, centroids):
    '''
    :param x: 某一个ground truth的w,h
    :param centroids:  anchor的w,h的集合[(w,h),(),...]，共k个
    :return: 单个ground truth box与所有k个anchor box的IoU值集合
    '''
    IoUs = []
    w, h = x  # ground truth的w,h
    for centroid in centroids:
        c_w, c_h = centroid  # anchor的w,h
        if c_w >= w and c_h >= h:  # anchor包围ground truth
            iou = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:  # anchor宽矮
            iou = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:  # anchor瘦长
            iou = c_w * h / (w * h + c_w * (c_h - h))
        else:  # ground truth包围anchor     means both w,h are bigger than c_w and c_h respectively
            iou = (c_w * c_h) / (w * h)
        IoUs.append(iou)  # will become (k,) shape
    return np.array(IoUs)

def avg_IOU(X, centroids):
    '''
    :param X: ground truth的w,h的集合[(w,h),(),...]
    :param centroids: anchor的w,h的集合[(w,h),(),...]，共k个
    '''
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        sum += max(IOU(X[i], centroids))  # 返回一个ground truth与所有anchor的IoU中的最大值
    return sum / n  # 对所有ground truth求平均

def write_anchors_to_file(centroids, X, anchor_file, input_shape, yolo_version):
    '''
    :param centroids: anchor的w,h的集合[(w,h),(),...]，共k个
    :param X: ground truth的w,h的集合[(w,h),(),...]
    :param anchor_file: anchor和平均IoU的输出路径
    '''
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    if yolo_version == 'yolov2':
        for i in range(anchors.shape[0]):
            # yolo中对图片的缩放倍数为32倍，所以这里除以32，
            # 如果网络架构有改变，根据实际的缩放倍数来
            # 求出anchor相对于缩放32倍以后的特征图的实际大小（yolov2）
            anchors[i][0] *= input_shape / 32.
            anchors[i][1] *= input_shape / 32.
    elif yolo_version == 'yolov3':
        for i in range(anchors.shape[0]):
            # 求出yolov3相对于原图的实际大小
            anchors[i][0] *= input_shape[0]
            anchors[i][1] *= input_shape[1]
    else:
        print("the yolo version is not right!")
        exit(-1)

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))
    avg_iou = avg_IOU(X, centroids)
    f.write('%f\n' % (avg_iou))
    print()
    return avg_iou

# X:一共有多少个需要聚类的宽高,tentor(4000*2)
# centroids:随机找的9个聚类中心
# eps: eps = 0.005
# anchor_file: 聚类出来的9个anchor的存放地址
# input_shape  图片的原始维度,例如:416*416
# yolo_version yolov3
def kmeans(X, centroids, eps, anchor_file, input_shape, yolo_version):
    N = X.shape[0]  # gr ound truth的个数
    iterations = 0
    print("centroids.shape", centroids)
    k, dim = centroids.shape  # anchor的个数k以及w,h两维，dim默认等于2
    prev_assignments = np.ones(N) * (-1)  # 对每个ground truth分配初始标签
    iter = 0
    old_D = np.zeros((N, k))  # 初始化每个ground truth对每个anchor的IoU

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)  得到每个ground truth对每个anchor的IoU

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))  # 计算每次迭代和前一次IoU的变化值

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)  # 将每个ground truth分配给距离d最小的anchor序号

        if (assignments == prev_assignments).all():  # 如果前一次分配的结果和这次的结果相同，就输出anchor以及平均IoU
            print("Centroids = ", centroids)
            input_shape = [416, 416]
            avg_iou = write_anchors_to_file(centroids, X, anchor_file, input_shape, yolo_version)
            return avg_iou

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)  # 初始化以便对每个簇的w,h求和
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]  # 将每个簇中的ground truth的w和h分别累加
        for j in range(k):  # 对簇中的w,h求平均
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1)

        prev_assignments = assignments.copy()
        old_D = D.copy()

if __name__ == '__main__':
    # mask = make_416_image("data/imgs/000001.jpg")
    # mask.show()
    with open('data/datas.txt','r') as f:
        datas = f.readlines()
        annotation_dims = []
        for data in datas:
            data_list = data[:-1].split(" ")[1:]
            for i,d in enumerate(data_list):
                if (i+1) % 5 == 1:
                    temp = []
                elif (i+1) % 5 == 4:
                    temp.append(int(d)/416)
                elif (i + 1) % 5 == 0:
                    temp.append(int(d)/416)
                    annotation_dims.append(temp)
    X = np.array(annotation_dims)
    centroids = X[:9]
    anchor_file = os.path.join('data', 'anchors.txt')
    kmeans(X,centroids,0.005,anchor_file,(416,416),'yolov3')