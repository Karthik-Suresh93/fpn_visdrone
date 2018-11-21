import random
import numpy as np
import xml.dom.minidom
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
MAX_OBJ_PER_IMAGE = 10
def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  max(b1_x1, b2_x1)
    inter_rect_y1 =  max(b1_y1, b2_y1)
    inter_rect_x2 =  min(b1_x2, b2_x2)
    inter_rect_y2 =  min(b1_y2, b2_y2)
    #Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width > 0 and inter_height > 0:#strong condition
        inter_area = inter_width * inter_height
        #Union Area
        b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area)
    else:
        iou = 0
    return iou

def aug_txt(img_idx, txt_path='./', img_path='./', image_save_path="./", txt_save_path ="./"):
    class_name = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
    img = Image.open(img_path+img_idx+".jpg")
    width, height = img.size
    fin = open(txt_path+img_idx+'.txt', 'r')
    #get all the gts
    bboxes = []
    lines = []
    for l in fin.readlines():
        l = l.split(',')
        class_id = int(l[5])
        if class_id  not in [0,11]:
            #print(class_id)
            lines.append(l)
            bboxes.append([int(l[0]), int(l[1]),int(l[0])+int(l[2])-1, int(l[1])+int(l[3])-1])

    fin.close()
     
    #generate new gt
    specific_class_idxs = [9]       #index of pedestrian, awning-tricycle [1,8]  ###CHANGE THIS FOR OTHER CLASS IDS... PROBABLY PEOPLE AND CARS
    threshold = 0.15
    sample_num_per_sample = 1        #number of patches of objects to be added... originally 2
    with open(txt_save_path+img_idx+'.txt', 'a+') as fout:
        n_obj=0
        for line in lines:
            class_name = int(line[5])
            assert class_name < 10
            #print(class_name)
            #print (class_name)       # if class_name  not in [0,11]: class_name = class_name - 1 ;remove ignored_regions and others
            if class_name in specific_class_idxs and n_obj<MAX_OBJ_PER_IMAGE:

                n_obj+=1
                bbox_left, bbox_top, bbox_width, bbox_height = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                for i in range(sample_num_per_sample):
                    new_bbox_left = random.randint(0, width-bbox_width)
                    new_bbox_top = random.randint(0,  height-bbox_height)
                    bbox1 =  [new_bbox_left, new_bbox_top, new_bbox_left+bbox_width-1, new_bbox_top+bbox_height-1]
                    ious = [bbox_iou(bbox1, bbox) for bbox in bboxes]
                    #print("max IOUS: ", max(ious))
                    if max(ious) <= threshold:
                        #write into txt file
                        fout.write(str(bbox1[0])+','+str(bbox1[1])+','+str(bbox_width)+','+str(bbox_height)+','+'0,'+str(class_name)+',0,0'+'\n')
                        #update bboxes list
                        bboxes.append(bbox1)
                        #update image
                        region=img.crop( (bbox_left, bbox_top, bbox_left+bbox_width-1, bbox_top+bbox_height-1))
                        img.paste(region, ( bbox1[0], bbox1[1]) )
                        print("saving image:", img_idx)
                        img.save(image_save_path+img_idx+"_.jpg")

def txt2xml(img_idx, xml_path = './', txt_path='./',img_path='./'):
    class_name = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
    fin = open(txt_path+img_idx+'.txt', 'r')
    img = Image.open(img_path+img_idx+".jpg")
    xml_name = xml_path+img_idx+'.xml'
    with open(xml_name, 'w') as fout:
        fout.write('<annotation>'+'\n')
        
        fout.write('\t'+'<folder>VOC2007</folder>'+'\n')
        fout.write('\t'+'<filename>'+img_idx+'.jpg'+'</filename>'+'\n')
        
        fout.write('\t'+'<source>'+'\n')
        fout.write('\t\t'+'<database>'+'VisDrone2018 Database'+'</database>'+'\n')
        fout.write('\t\t'+'<annotation>'+'VisDrone2018'+'</annotation>'+'\n')
        fout.write('\t\t'+'<image>'+'flickr'+'</image>'+'\n')
        fout.write('\t\t'+'<flickrid>'+'Unspecified'+'</flickrid>'+'\n')
        fout.write('\t'+'</source>'+'\n')
        
        fout.write('\t'+'<owner>'+'\n')
        fout.write('\t\t'+'<flickrid>'+'Haipeng Zhang'+'</flickrid>'+'\n')
        fout.write('\t\t'+'<name>'+'Haipeng Zhang'+'</name>'+'\n')
        fout.write('\t'+'</owner>'+'\n')
        
        fout.write('\t'+'<size>'+'\n')
        fout.write('\t\t'+'<width>'+str(img.size[0])+'</width>'+'\n')
        fout.write('\t\t'+'<height>'+str(img.size[1])+'</height>'+'\n')
        fout.write('\t\t'+'<depth>'+'3'+'</depth>'+'\n')
        fout.write('\t'+'</size>'+'\n')
        
        fout.write('\t'+'<segmented>'+'0'+'</segmented>'+'\n')

        for line in fin.readlines():
            line = line.split(',')
            assert int(line[5]) <10, int(line[5])
            fout.write('\t'+'<object>'+'\n')
            fout.write('\t\t'+'<name>'+class_name[int(line[5])]+'</name>'+'\n')
            fout.write('\t\t'+'<pose>'+'Unspecified'+'</pose>'+'\n')
            fout.write('\t\t'+'<truncated>'+line[6]+'</truncated>'+'\n')
            fout.write('\t\t'+'<difficult>'+str(int(line[7]))+'</difficult>'+'\n')
            fout.write('\t\t'+'<bndbox>'+'\n')
            fout.write('\t\t\t'+'<xmin>'+line[0]+'</xmin>'+'\n')
            fout.write('\t\t\t'+'<ymin>'+line[1]+'</ymin>'+'\n')
            # pay attention to this point!(0-based)
            fout.write('\t\t\t'+'<xmax>'+str(int(line[0])+int(line[2])-1)+'</xmax>'+'\n')
            fout.write('\t\t\t'+'<ymax>'+str(int(line[1])+int(line[3])-1)+'</ymax>'+'\n')
            fout.write('\t\t'+'</bndbox>'+'\n')
            fout.write('\t'+'</object>'+'\n')
        fout.write('</annotation>')
        
def read_single_img(img_idx):
    imgfile = img_idx +'_.jpg' 
    xmlfile = img_idx + '.xml'
    img = Image.open("../../JPEGImages/"+imgfile)
    DomTree = xml.dom.minidom.parse("../../VisDrone2018-DET-train/Annotations_XML_Aug/"+xmlfile)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename')
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data
        #print(objectname)
        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox:
            if objectname == 'pedestrian':
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                  
#                draw = ImageDraw.Draw(img)
#                # green color for rectangle
#                draw.rectangle([x1,y1,x2,y2],outline="green")
#                # red color for text
#                draw.text((x1,y1),objectname,fill=(255,0,0,255))
#    img.show()

if __name__ == '__main__':
    import os
    img_idx_all = [imdx.split(".")[0] for imdx in os.listdir("./JPEGImages_10_classes_normal",)]
    for img_idx in img_idx_all:                                                                #'9999999_00231_d_0000098' #for img in os.listdir("../")
        aug_txt(img_idx, txt_path='./annotations_10_classes/', img_path='./JPEGImages_10_classes_normal/', txt_save_path = "./Annotations_Aug_9_Text/", image_save_path = "./JPEGImages_Aug_9/")
        txt2xml(img_idx, xml_path = './Annotations_Aug_9_XML/', txt_path='./Annotations_Aug_9_Text/',img_path='./JPEGImages_10_classes_normal/')
