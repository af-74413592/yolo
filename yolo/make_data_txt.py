import xml.etree.cElementTree as et
import os
import math
import queue
import json

# count = queue.PriorityQueue()
# for i in range(30):
#     count.put(i)
count = 0
class_count = {}

xml_dir = 'data/anno'
xml_filenames = os.listdir(xml_dir)
datas = []
for xml_filename in xml_filenames:
    xml_filename_path = os.path.join(xml_dir,xml_filename)
    tree = et.parse(xml_filename_path)
    root = tree.getroot()
    filename = root.find('filename')
    names = root.findall('object/name')
    boxs = root.findall('object/bndbox')
    data = []
    data.append(filename.text)
    for name,box in zip(names,boxs):
        try:
            cls = class_count[name.text]
        except:
            #cls = count.get()
            cls = count
            count += 1
        class_count[name.text] = cls
        data.append(str(cls))
        x1,y1,x2,y2 = box
        cx = math.floor((int(x1.text)+int(x2.text))/2)
        cy = math.floor((int(y1.text)+int(y2.text))/2)
        w = int(x2.text) - int(x1.text)
        h = int(y2.text) - int(y1.text)
        data.append(str(cx))
        data.append(str(cy))
        data.append(str(w))
        data.append(str(h))
    data_text = " ".join(data) +'\n'
    datas.append(data_text)
with open("data/datas.txt",'w') as f:
    f.writelines(datas)
json.dump(class_count,open('data/class_count.json','w'))