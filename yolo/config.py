import math
DATA_WIDTH = 416
DATA_HEIGHT = 416

CLASS_NUM = 20

Centroids =  [[0.3159778,  0.65469678],
 [0.78684117, 0.66489416],
 [0.15067512, 0.28257979],
 [0.05449378, 0.05489684],
 [0.05190121, 0.14707144],
 [0.10354336, 0.09357196],
 [0.54628594, 0.28921297],
 [0.22210977, 0.10480571],
 [0.01793931, 0.00979297]]
antors_list = []
for i in Centroids:
    w = math.floor(i[0] * 416)
    h = math.floor(i[1] * 416)
    antors_list.append([w,h])
antors_list = sorted(antors_list, key=lambda x: x[0]*x[1], reverse=True)
antors = {13:antors_list[:3],26:antors_list[3:6],52:antors_list[6:9]}
ANTORS_AREA={
    13: [x*y for x,y in antors[13]],
    26: [x*y for x,y in antors[26]],
    52: [x*y for x,y in antors[52]]
}

