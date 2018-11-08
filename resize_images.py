#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:39:11 2018
@author: karthik
"""
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import os
from PIL import Image
from scipy import ndimage


import csv

RESIZE_TO = (3000, 3000)
PATH_IMAGE = "./data/VOCdevkit2007/VOC2007/JPEGImages_10_Class_normal"
PATH_annot = "./data/VOCdevkit2007/VOC2007/annotations_10_classes/"
PATH_SAVE_ANNOT = "./data/VOCdevkit2007/VOC2007/Annotations_"+str(RESIZE_TO[0])+"_txt"
PATH_SAVE_IMAGE = "./data/VOCdevkit2007/VOC2007/JPEGImages_"+str(RESIZE_TO[0])+"/"

count = 0
for image_name in (os.listdir(PATH_IMAGE)):
    count+=1
    print("done with:", count)
    img_size = cv2.imread(PATH_IMAGE+"/"+image_name).shape[:2]
    #print(img_size)
    image = ndimage.imread(PATH_IMAGE+"/"+image_name, mode="RGB")
    with open(PATH_annot + image_name.split(".")[0]+ ".txt") as f:
        lines = f.readlines()
        bb_list=[]
        scto = []
        scto_dict = {}
        for line in lines:
            xmin, ymin, w, h = line.split(",")[0:4]
            [score, category, truncation, occlusion] = line.split(",")[4:8]
            scto.append([score,category, truncation, occlusion])
            xmin=int(xmin)
            ymin = int(ymin)
            w = int(w)
            h = int(h)
            
            xmax = xmin + w
            ymax = ymin + h
            bb_list.append(ia.BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax))
        bbs = ia.BoundingBoxesOnImage(bb_list, shape=image.shape)
        scto_dict[image_name] = scto
    image_rescaled = ia.imresize_single_image(image, RESIZE_TO)
    bbs_rescaled = bbs.on(image_rescaled)
    image_rescaled = Image.fromarray(image_rescaled, "RGB")
    image_rescaled.save(PATH_SAVE_IMAGE+image_name)

    
    with open(PATH_SAVE_ANNOT+"/"+image_name.split(".")[0]+".txt", "a+") as f2:
        for idx, bbs in enumerate(bbs_rescaled.bounding_boxes):
            full_line = [int(bbs.x1), int(bbs.y1), int(bbs.x2)-int(bbs.x1), int(bbs.y2)-int(bbs.y1),int(scto_dict[image_name][idx][0]), int(scto_dict[image_name][idx][1]), int(scto_dict[image_name][idx][2]), int(scto_dict[image_name][idx][3])]
            wr = csv.writer(f2)
            wr.writerow(full_line)
print("total number of files are:", count)
## Draw image before/after rescaling and with rescaled bounding boxes
# image_bbs = bbs.draw_on_image(image,thickness=2)
# image_rescaled_bbs = bbs_rescaled.draw_on_image(image_rescaled, thickness=2)

# img_original =Image.fromarray(image_bbs, "RGB")
# img_rescaled =Image.fromarray(image_rescaled_bbs, "RGB")
# #img.save("name.jpg")
# img_original.show()
# image_rescaled.show()