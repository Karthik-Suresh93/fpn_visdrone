#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 19:34:51 2018
@author: karthik
"""
import cv2
import os
import csv
PATH_IMAGE = "./JPEGImages/"
PATH_annot = "./annotations_car_only_txt/"
PATH_IMAGE_4 = "./JPEGImages_split_5000_0.15/"


PATH_ANNOT_4 = "./Annotations_split_5000_0.15/"

def divide_img_into_two_vertically(img, overlap_w):

    """
    overlap_w: amount of overlap expressed as fraction of half the width of the image. or width of half a horizontally cut portion
    """
    height, width = img.shape[:2]    
    # Let's get the starting pixel coordiantes (top left of cropped top)
    start_row, start_col = int(0), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_row, end_col = int(height), int(width*0.5 + overlap_w*width*0.5)
    cropped_left = img[start_row:end_row , start_col:end_col]       
    # Let's get the starting pixel coordiantes (top left of cropped bottom)
    start_row, start_col = int(0), int(width*0.5)   #next half is not affected by overlap
    # Let's get the ending pixel coordinates (bottom right of cropped bottom)
    end_row, end_col = int(height), int(width)
    cropped_right = img[start_row:end_row , start_col:end_col]    
    return(cropped_left, cropped_right)

def divide_img_into_two_horizontally(img, overlap_h):
    
    height, width = img.shape[:2]   
    # Let's get the starting pixel coordiantes (top left of cropped top)
    start_row, start_col = int(0), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped top)
    end_row, end_col = int(height*0.5 + overlap_h*0.5*height), int(width)
    cropped_left = img[start_row:end_row , start_col:end_col]    
    
    # Let's get the starting pixel coordiantes (top left of cropped bottom)
    start_row, start_col = int(height*0.5), int(0)
    # Let's get the ending pixel coordinates (bottom right of cropped bottom)
    end_row, end_col = int(height), int(width)
    cropped_right = img[start_row:end_row , start_col:end_col]    
    return(cropped_left, cropped_right)
    
count_separated_anno=0
count_all_anno=0

def divide_img_into_four_overlap(img, overlap_w = 0.15, overlap_h = 0.15):

    l,r = divide_img_into_two_vertically(img, overlap_w)
    q1, q2 = divide_img_into_two_horizontally(l, overlap_h)
    q3, q4 = divide_img_into_two_horizontally(r, overlap_h)

    return q1,q2,q3,q4

# image_name =  (str(os.listdir(PATH_IMAGE)[1]))
# image =  cv2.imread(PATH_IMAGE+ image_name)
# q1,q2,q3,q4 = divide_img_into_four_overlap(image, overlap_w = 0, overlap_h = 0)
# cv2.imwrite("./q1.jpg", q1)
# cv2.imwrite("./q2.jpg", q2)
# cv2.imwrite("./q3.jpg", q3)
# cv2.imwrite("./q4.jpg", q4)

def adjust_annotations(image_name, img, overlap_h=0.15, overlap_w=0.15):
    
    with open(PATH_annot + image_name.split(".")[0]+".txt") as f:
        lines = f.readlines()
        for line in lines:
            bbox = line.split(",")[0:4]
            [score, category, truncation, occlusion] = line.split(",")[4:8]
            assert int(category)<10, image_name
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2]) +xmin
            ymax = int(bbox[3]) +ymin

            assert xmin<xmax, "xmin>xmax!"
            assert ymin<ymax, "ymin>ymax!"

            h,w = img.shape[:2]
            overlap_adjust_w = 0.5*overlap_w*w
            overlap_adjust_h = 0.5*overlap_h*h
            #quadrant 1: left, top
            #annotations remain the same
            if ymax < h/2+overlap_adjust_h and xmax < w/2+overlap_adjust_w:
                
                full_line = [xmin, ymin, xmax-xmin, ymax-ymin, int(score), int(category),int(truncation), int(occlusion)]
                with open(PATH_ANNOT_4+image_name.split(".")[0]+"_1"+".txt", "a+") as f1:
                    wr = csv.writer(f1)
                    wr.writerow(full_line)

            #quadrant 2: left, bottom

            elif xmax < w/2+overlap_adjust_w and ymin >h/2:
                ymax = ymax - h/2
                ymin = ymin - h/2
                full_line = [xmin, ymin, xmax-xmin, ymax-ymin, int(score), int(category),int(truncation), int(occlusion)]
                with open(PATH_ANNOT_4+image_name.split(".")[0]+"_2"+".txt", "a+") as f2:
                    wr = csv.writer(f2)
                    wr.writerow(full_line)
            #quadrant 3: right, top

            elif xmin >w/2 and ymax < h/2 +overlap_adjust_h:
                xmin = xmin - w/2
                xmax = xmax - w/2
                full_line = [xmin, ymin, xmax-xmin, ymax-ymin, int(score), int(category),int(truncation), int(occlusion)]
                with open(PATH_ANNOT_4+image_name.split(".")[0]+"_3"+".txt", "a+") as f3:
                    wr = csv.writer(f3)
                    wr.writerow(full_line)

            #quadrant 4: right, bottom
            elif ymin > h/2 and xmin > w/2:
                ymin = ymin - h/2
                ymax = ymax - h/2
                xmin = xmin - w/2
                xmax = xmax - w/2
                full_line = [xmin, ymin, xmax-xmin, ymax-ymin, int(score), int(category),int(truncation), int(occlusion)]
                with open(PATH_ANNOT_4+image_name.split(".")[0]+"_4"+".txt", "a+") as f4:
                    wr = csv.writer(f4)
                    wr.writerow(full_line)

            else:
                #print ("bb is in none of the quadrants!")
                pass



if __name__ == "__main__":
    for image_name in os.listdir(PATH_IMAGE):
        image = cv2.imread(PATH_IMAGE+image_name)
        #print(image.shape)
        # cs, ca= adjust_annotations(image_name, image,overlap=0.40)
    #    l,r = divide_img_into_two(image)
    #    one, two = divide_img_into_two(l)
    #    three, four = divide_img_into_two(r)
    # one,two,three,four = divide_img_into_four_with_overlap(image, overlap_amount = 0.40)
        o_h = 0.15; o_w = 0.15 #SET OVERLAPS ALONG HEIGHT AND WIDTH RESPECTIVELY...
        q1,q2,q3,q4 = divide_img_into_four_overlap(image, overlap_h = o_h, overlap_w = o_w)
        cv2.imwrite(PATH_IMAGE_4+image_name.split(".")[0]+"_1"+".jpg", q1)
        cv2.imwrite(PATH_IMAGE_4+image_name.split(".")[0]+"_2"+".jpg", q2)
        cv2.imwrite(PATH_IMAGE_4+image_name.split(".")[0]+"_3"+".jpg", q3)
        cv2.imwrite(PATH_IMAGE_4+image_name.split(".")[0]+"_4"+".jpg", q4)
        adjust_annotations(image_name, image, overlap_h = o_h, overlap_w = o_w)
    
#import matplotlib.pyplot as plt
#plt.plot(list(range(12)), [class_list_original.count(i) for i in range(12)], color="b")
#plt.plot(list(range(12)), [class_list_40.count(i) for i in range(12)], color="g")
#
#plt.show()
