from __future__ import print_function
import os
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
import xml.dom.minidom
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw




root_dir = "./"
#root_dir = "/home/ksuresh/fpn.pytorch-master/data/uavdt/data/VOCdevkit2007/UAV2017/"
#ImgPath = root_dir+'JPEGImages_split_4_overlap_0/' 
AnnoPath = root_dir+'Annotations/' #_10_classes_xml/'

scale = []
area = []
aspect_ratio = []
def scale_and_aspect(img_idx):
    # imgfile = ImgPath + img_idx+'.jpg'
    xmlfile = AnnoPath + img_idx + '.xml'
    #imgfile = '0000097_09823_d_0000051_det.jpg'   
    #xmlfile = '0000097_09823_d_0000051.xml'   
    # img = Image.open(imgfile)
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename')
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    #bboxes.append(len(objectlist))    
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data

        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox:
            #if objectname == 'pedestrian':
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            #scale.append(get_scale([x1,y1,x2,y2]))
            if y2 ==y1:
                y2+=1
                print("fault...")
            if x2 ==x1:
                x2+=1
                print("faulty")
            aspect_ratio.append(get_aspect_ratio([x1,y1,x2,y2]))
            area.append((x2-x1)*(y2-y1))

    return aspect_ratio, area#, scale

def get_scale(bbox):
    pass

def get_aspect_ratio(bbox):
    return((float(bbox[2])-bbox[0])/(float(bbox[3])-bbox[1]))

def plot(aspect_ratio, area, scale):
    import matplotlib.pyplot as plt
    plt.hist(np.array(aspect_ratio), bins = 75, range = (0,5))
    # plt.xlabel("number of points")
    # plt.ylabel("aspect ratio")
    plt.show()
    plt.hist(area, bins = 75, range = (0,25000))
    plt.show()

if __name__ == '__main__':
    for img_idx in os.listdir(AnnoPath):
        img_idx = img_idx.split(".")[0]
        #9999966_00000_d_0000093_2': weird
    #args = parser.parse_args()

        asr, area= scale_and_aspect(img_idx)
    # with open("./aspect_ratio.txt", "w") as f:
    #     print(asr, file = f)
    #     print(area)
    plot(asr, area, [])
