import os
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
import xml.dom.minidom
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import csv



root_dir = "./"
ImgPath = root_dir+'JPEGImages/' #_10_classes_normal/' #"./JPEGImages_split_4_overlap_0/" #'/VOC_ORIGINAL/VOCdevkit/VOC2007/JPEGImages/' 
AnnoPath = root_dir+'Annotations/' #_10_classes_xml/' #'Annotations_4_overlap_0_xml/' #'/VOC_ORIGINAL/VOCdevkit/VOC2007/Annotations/'
AnnoPath_txt = root_dir+'annotations_car_only_txt/'

def txt_to_image_single(img_idx):
    #imgfile = ImgPath + img_idx+'.jpg' 
    xmlfile = AnnoPath + img_idx + '.xml'
    #imgfile = '0000097_09823_d_0000051_det.jpg'   
    #xmlfile = '0000097_09823_d_0000051.xml'   
    #img = Image.open(imgfile)
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename')
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    #bboxes.append(len(objectlist))
    full_file = []
    
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
            full_file.append([str(x1), str(y1), str(x2-x1), str(y2-y1),"1", "1", "0", "0"])  #score =1, class = , only car, truncation and occluasion is also 1
    
    with open(AnnoPath_txt+img_idx+".txt", "w") as fout:
        wr = csv.writer(fout)
        for line in full_file:
            wr.writerow(line)



if __name__ == '__main__':

    for img_idx in os.listdir(AnnoPath):
        img_idx = img_idx.split(".")[0]
        print(img_idx)
        txt_to_image_single(img_idx)
    
    #parser = argparse.ArgumentParser('main')
    #parser.add_argument('img_idx')
    #img_idx = '0000001_03999_d_0000007'  #9999966_00000_d_0000093_2': weird
    #args = parser.parse_args()
    #read_single_img(img_idx)
    
    # train_list = "../../data/Merge_VisDrone2018/ImageSets/Main/train.txt"
    # #read_all_images(train_list)
    # #get_bbox()
    # #print(bboxes.keys())

    # img_dir = "../../Data/VisDrone2018-DET-train/images/"
    # img_size = 1050
    #size=2000,RGB:95.003518834106174, 96.386738363931443, 92.885945304899025
    #size=1400,RGB:95.035268631239916, 96.41846471675818, 92.917909987495165
    #size=1050,RGB:95.060276139104829, 96.44352133232799, 92.942862747769652
    #get_mean_rgb(img_dir,img_size)

