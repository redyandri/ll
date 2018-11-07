from __future__ import division
#from __future__ import print_function
import sys
from scapy.utils6 import construct_source_candidate_set

from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tflow
from tensorflow.python import pywrap_tensorflow
import csv
import string
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.patches as mpatches
import os
import re
from termcolor import colored
import glob
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image
from collections import Counter
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
import glob, os, sys
from os.path import basename
import matplotlib.pyplot as plt
import math
from shutil import copytree
import cv2
from sklearn.model_selection import KFold
from random import randint
from skimage import io
from skimage import transform as tf
import Augmentor
from PIL import Image
import PIL.ImageOps





class victorinox(object):
    def __init__(self):
        return

    def iterate_files(self, folder, receiver=[]):
        f = os.listdir(folder)
        for i in f:
            ff = os.path.join(folder, i)
            if os.path.isfile(ff):
                # print str(ff)
                receiver.append(ff)
            else:
                self.iterate_files(ff, receiver)
        return receiver

    def get_file_name(self, file_path):
        paths=[]
        p = re.compile("[a-zA-Z]+[a-zA-Z_\-]+[a-zA-Z]+")
        head,tail=os.path.split(file_path)
        name=tail.split(".")[0]
        # name=p.findall(name)[0]
        if name.startswith("text_"):
            name=name[5:]
        if name.startswith("text-with-special-symbols_"):
            name=name[26:]
        if name.endswith("_non-text"):
            name=name[0:name.index("_non-text")]
        return name

    def fuse_features(self,csvs=[]):
        df = pd.read_csv(csvs[0])
        dfs = np.empty(shape=(df.shape[0], 1))
        rows = []
        name = []
        for i in range(0, len(csvs)):
            df = pd.read_csv(csvs[i])
            df = np.array(df)
            col_len = df.shape[1]
            name = df[:, -1]
            sub = df[:, 0:col_len - 1]
            dfs = np.hstack((dfs, sub))
        name = np.reshape(name, (name.shape[0], 1))
        dfs = np.hstack((dfs, name))
        return dfs[:, 1:]  # exclude first column that contains init values of np.empty

    # def select_hifreq_class(self,class_to_omit,csv_src,csv_dest):
    #     p = csv_src
    #     df = pd.read_csv(p)
    #     minor = class_to_omit
    #         # [
    #         # "list-item",
    #         # "reference-list_item"
    #         # "pseudo-code",
    #         # "logo",
    #         # "synopsis",
    #         # "advertisement",
    #         # "map",
    #         # "publication-info",
    #         # "highlight",
    #         # "reader-service",
    #         # "keyword_heading_and_body",
    #         # "seal",
    #         # "membership",
    #         # "diploma",
    #         # "correspondence",
    #         # "announcement",
    #         # "abstract_heading_and_body"]
    #     high_freq = np.array([row for row in df.itertuples(index=False) if row[-crossval1.old] not in minor])  #skip rows whose class is minor or to omit
    #     with open(csv_dest, "wb") as dest:
    #         writer = csv.writer(dest)
    #         for row in high_freq[:]:
    #             row = row.tolist()
    #             writer.writerow(row)
    #         dest.close()

    def print_confusion_matrix(self,ground_truths,predictions,class_dict={}):
        predictions=[class_dict[p] for p in predictions]
        ground_truths = [class_dict[g] for g in ground_truths]
        set1 = np.vstack((predictions, ground_truths))
        class_set = np.unique(set1) #get set of labels
        sorted_set = sorted(class_set)
        sorted_set = np.array(sorted_set)
        cm = confusion_matrix(ground_truths, predictions)
        h_labels = np.reshape(sorted_set, (1, sorted_set.shape[0]))
        cm = np.vstack((h_labels, cm))
        added_sort = np.concatenate((["*"], sorted_set))
        v_labels = np.reshape(added_sort, (added_sort.shape[0], 1))
        cm = np.hstack((v_labels, cm))
        cols = [i[0] for i in enumerate((cm[0, :]))]
        pt = PrettyTable(field_names=cols)
        for row in cm[1:, :]:
            total=sum([int(val) for val in row[1:]])
            if total==0:
                total=1
            i=1
            for r in row[1:]:
                row[i]=round(int(r)/total,2)
                i+=1
            pt.add_row(row.tolist())
        print pt
        for i in cols:
            if i>0:
                print "%d : %s" %(i,cm[0,i])

    def binarize(self, dataset_path):
        images=[]
        images=self.iterate_files(dataset_path,images)
        images=filter(lambda x:x.endswith(".jpg"),images)
        c=1
        print "%d images found" %len(images)
        for img in images:
            try:
                bashCommand = "python tool/anyBaseOCR-nlbin.py "+img
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                print "%d   %s is binarized"%(c, os.path.split(img)[1])
                c+=1
            except Exception,e:
                print "Error:"+str(e)

    def iterate_zones_coord_from_dataset(self,dataset_path):
        images = []
        images = self.iterate_files(dataset_path, images)
        images = filter(lambda x: x.endswith(".xml"), images)
        class_coords = []
        for img in images:
            try:
                tree = ET.parse(img)
                root = tree.getroot()
                img_f = ""
                main_children = root.getchildren()
                for x in root:
                    # print x.tag
                    if str(x.tag).endswith("Page"):
                        img_f = x.attrib["imageFilename"]
                        img_f=img_f.split("/")[-1] #get only file name
                        img_f=self.find_image_path(img_f,dataset_path) #get absolute path of img
                        children = x.getchildren()
                        for y in children:
                            if "Region" in y.tag:
                                class_type = dict(y.attrib).values()[0]
                                coord_tag = y.getchildren()
                                coords = coord_tag[0].attrib["points"]
                                class_coord = [img_f, class_type, coords.split(" ")]
                                class_coords.append(class_coord)
            except Exception,e:
                print "Error:%s"%str(e)

        return class_coords

    def extract_zones_from_xml(self,xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_f = ""
        class_coords = []
        main_children = root.getchildren()
        mistagged_xml=""
        for x in root:
            # print x.tag
            try:
                if str(x.tag).endswith("Page"):
                    img_f = x.attrib["imageFilename"]
                    w,h=0,0
                    if(x.attrib["imageWidth"]!=""):
                       w=int(x.attrib["imageWidth"])
                    if(x.attrib["imageHeight"]!=""):
                        h=int(x.attrib["imageHeight"])
                    children = x.getchildren()
                    for y in children:
                        if "Region" in y.tag:
                            tag = str(y.tag)
                            tag = tag.split("}")[1]  # remove {http://blabla} from {http://blabla}textregion
                            class_type = tag[0:tag.index("Region")]
                            if class_type == "Text":
                                class_type = y.attrib["type"]
                            region_id = y.attrib["id"].split("_")[1]
                            class_type = class_type + "_" + region_id
                            coord_tag = y.getchildren()
                            coords = coord_tag[0].attrib["points"]
                            class_coord = [img_f, class_type, coords.split(" "),[w,h]]
                            class_coords.append(class_coord)
            except Exception,e:
                print "Error XML reading:%s"%str(e)
                mistagged_xml=xml_path
                continue
        return class_coords,mistagged_xml


    def crop_and_convert_tif_zones(self,src_dataset,dest_dataset):
        self.imitate_image_filetree(src_dataset,dest_dataset)     #imitate file tree of src to dest zone folder
        count = 1
        not_found=[]
        for dirpath,dirnames,filenames in os.walk(src_dataset):
            if dirpath.endswith("page"):
                for xml_file in filenames:
                    xml_file_path=os.path.join(dirpath,xml_file)
                    zones=[]
                    mistagged_xml =""
                    try:
                        zones, mistagged_xml = self.extract_zones_from_xml(xml_file_path)  # list of list [jpg_path,classtyp,(4 points coordinate]]
                        zone_seq_per_img=1
                        for zone in zones:
                            img = "/".join((zone[0]).split("/")[-2:]) #'src/jpg/image.jpg.xml'
                            temp_dir=dirpath.replace("page","") #'src/a/b/c/page -> src/a/b/c/'
                            img = img.replace(".xml", "") #'src/image.jpg.xml'->#'src/image.jpg
                            img = os.path.join(temp_dir, img)
                            zone_type = zone[1]
                            zone_coord = zone[2]
                            c = []
                            for z in zone_coord:
                                xy = z.split(",")
                                z = (int(xy[0]), int(xy[1]))  #tuple of coord [(0,0),(2,2),(),()]
                                c.append(z)
                            if os.path.exists(img):
                                xml_file_name = str(xml_file).replace(".jpg.xml", "")
                                zone_name = xml_file_name + "_" + zone_type + ".jpg"  # ex:1_paragraph.jpg ,count is to prevent overwriting
                                dest_zone_path = "/".join(
                                    (img.split("/")[-3:-1]))  # 'folder/jpg'  omit the page file name
                                norm_crop = xml_file_name + "_" + zone_type + ".nrm.png"
                                norm_crop_path = os.path.join(dest_dataset, dest_zone_path, norm_crop)
                                dest_zone_path = os.path.join(dest_dataset, dest_zone_path, zone_name)
                                page = Image.open(img)  # main page to zonify
                                zone_img = page.crop((c[0][0], c[0][1], c[2][0], c[2][1]))  # topleft and bottomright
                                zone_img.save(dest_zone_path)
                                # now convert cropped to .tif
                                dest_tif_zone = str(dest_zone_path).replace(".jpg", ".tif")
                                bashCommand = "convert " + dest_zone_path + " " + dest_tif_zone
                                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                                output, error = process.communicate()
                                # now binarize it
                                print "CONVERT TO TIF: %s" % (zone_name)
                                count += 1
                    except Exception, e:
                        print "Error:" + str(e)
                        not_found.append(dest_zone_path+"-------------"+str(e))
                        continue



                        #break
                    #break
                #break
        print "%d zones created" %count
        print "not found:\n%s" %not_found
        with open("output.txt","wb") as result:
            result.write("Misstagged XMLs:\n")
            for n in not_found:
                result.write(n)
                result.write("\n")
        return

    def remove_cropped_jpg_zones(self,src_dataset,dest_dataset):
        self.imitate_image_filetree(src_dataset,dest_dataset)     #imitate file tree of src to dest zone folder
        count = 1
        not_found=[]
        for dirpath,dirnames,filenames in os.walk(src_dataset):
            if dirpath.endswith("page"):
                for xml_file in filenames:
                    xml_file_path=os.path.join(dirpath,xml_file)
                    zones=[]
                    mistagged_xml =""
                    try:
                        zones, mistagged_xml = self.extract_zones_from_xml(xml_file_path)  # list of list [jpg_path,classtyp,(4 points coordinate]]
                        zone_seq_per_img=1
                        for zone in zones:
                            img = "/".join((zone[0]).split("/")[-2:]) #'src/jpg/image.jpg.xml'
                            temp_dir=dirpath.replace("page","") #'src/a/b/c/page -> src/a/b/c/'
                            img = img.replace(".xml", "") #'src/image.jpg.xml'->#'src/image.jpg
                            img = os.path.join(temp_dir, img)
                            zone_type = zone[1]
                            zone_coord = zone[2]
                            c = []
                            for z in zone_coord:
                                xy = z.split(",")
                                z = (int(xy[0]), int(xy[1]))  #tuple of coord [(0,0),(2,2),(),()]
                                c.append(z)
                            if os.path.exists(img):
                                xml_file_name = str(xml_file).replace(".jpg.xml", "")
                                zone_name = xml_file_name + "_" + zone_type + ".jpg"  # ex:1_paragraph.jpg ,count is to prevent overwriting
                                dest_zone_path = "/".join(
                                    (img.split("/")[-3:-1]))  # 'folder/jpg'  omit the page file name
                                norm_crop = xml_file_name + "_" + zone_type + ".nrm.png"
                                norm_crop_path = os.path.join(dest_dataset, dest_zone_path, norm_crop)
                                dest_zone_path = os.path.join(dest_dataset, dest_zone_path, zone_name)
                                # now convert cropped to .tif
                                dest_tif_zone = str(dest_zone_path).replace(".jpg", ".tif")
                                if os.path.exists(dest_zone_path):
                                    os.remove(dest_zone_path)   #remove rgb crop : xmlname_header.jpg
                                    # os.remove(norm_crop_path) #remove norm crop: xmlname_header.nrm.png
                                    print "DELETED JPG CROP: %s" % (zone_name)
                                    count += 1
                    except Exception, e:
                        print "Error:" + str(e)
                        not_found.append(dest_zone_path+"-------------"+str(e))
                        continue
        print "%d zones created" %count
        print "not found:\n%s" %not_found
        with open("output.txt","wb") as result:
            result.write("Misstagged XMLs:\n")
            for n in not_found:
                result.write(n)
                result.write("\n")
        return


    #
    # def imitate_image_filetree(self,src_dataset,dest_dataset):
    #     for dirpath, dirnames, filenames in os.walk(src_dataset):
    #         if not dirpath.endswith("page"): #select only jpg folder containg images
    #             structure = os.path.join(dest_dataset, dirpath[len(src_dataset)+crossval1.old:])
    #             if not os.path.exists(structure):
    #                 os.mkdir(structure,0777)
    #                 print "Created: %s" %str(structure)
    def imitate_filetree(self,src_dataset="/a/b/c",dest_dataset="/a/b/c"):
        for dirpath, dirnames, filenames in os.walk(src_dataset):
            structure = os.path.join(dest_dataset, dirpath[len(src_dataset)+1:])
            if not os.path.isdir(structure):
                os.mkdir(structure,0777)
                print "CREATED FOLDER: %s" %str(structure)

    def find_image_path(self,name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)


    def get_classname_distribution(self,dataset_path,img_ext=".bin.png",name_splitter="_",start_classname_idx=-1,end_classname_idx=0):
        tool = victorinox()
        classes = []
        for dirpath, dirs, files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        crop=str(file).replace(img_ext,"")
                        crop_path = os.path.join(dirpath, crop)
                        if end_classname_idx==0:
                            class_type = str(crop).split(name_splitter)[start_classname_idx:]
                        else:
                            class_type = str(crop).split(name_splitter)[start_classname_idx:end_classname_idx]
                        class_type="_".join(class_type)
                        classes.append(class_type)
        c = Counter(classes)
        c = c.most_common()
        return c

    def generate_hifreq_dataset(self, src_dataset_path="/a/b/c", hifreq_dataset_path="/a/b/c",image_ext=".bin.png",filename_splitter="_",classname_idx=-1,min_occur=50):
        self.imitate_filetree(src_dataset=src_dataset_path,dest_dataset=hifreq_dataset_path)
        class_dist=self.get_classname_distribution(dataset_path=src_dataset_path)
        num_per_class=class_dist[len(class_dist)-1][1] #get least number from counter
        classname_enum=list(enumerate([x for (x,y) in class_dist]))
        class_dictionary={classname:id for id,classname in classname_enum}
        class_counter = {classname: 0 for id, classname in classname_enum}
        anonymous_classes=[]
        classes_of_hifreq_dataset=[]
        hifreq_class = [x for (x, y) in class_dist if y >= min_occur]
        low_freq = []
        high_freq = []
        class_num=0
        for dirpath, dirs, files in os.walk(src_dataset_path):
            if len(files)>0:
                for crop in files:
                    if crop.endswith(image_ext):
                        crop_path = os.path.join(dirpath, crop)
                        class_type = str(crop).split(filename_splitter)[classname_idx]
                        class_type=class_type[0:str(class_type).index(image_ext)]
                        if class_type == "":
                            anonymous_classes.append(crop_path)
                            continue
                        if class_type in hifreq_class:
                            class_id=class_dictionary[class_type]
                            hifreq_file_path=os.path.join(hifreq_dataset_path,dirpath[len(src_dataset_path)+1:],crop)
                            shutil.copyfile(crop_path,hifreq_file_path)
                            print "copy %s to %s" %(crop_path,hifreq_file_path)
                            row = [hifreq_file_path, class_id]
                            class_num += 1
                            high_freq.append(row)
                            class_counter[class_type] = class_num
                        else:
                            low_freq.append(class_type)
                            continue
        # np_classes_of_hifreq=np.array(high_freq)
        # np.savetxt(csv_hifreq_dataset,np_classes_of_hifreq, delimiter=" ",fmt="%s %s")
        print "%d CLASSES COPIED." %len(high_freq)
        return high_freq

    def generate_mini_dataset(self, src_dataset_path="/a/b/c", mini_dataset_path="/a/b/c",image_ext=".bin.png",filename_splitter="_",classname_idx=-1):
        self.imitate_filetree(src_dataset=src_dataset_path,dest_dataset=mini_dataset_path)
        class_dist=self.get_classname_distribution(dataset_path=src_dataset_path)
        num_per_class=class_dist[len(class_dist)-1][1] #get least number from counter
        classname_enum=list(enumerate([x for (x,y) in class_dist]))
        class_dictionary={classname:id for id,classname in classname_enum}
        class_counter = {classname: 0 for id, classname in classname_enum}
        anonymous_classes=[]
        classes_of_mini_dataset=[]
        for dirpath, dirs, files in os.walk(src_dataset_path):
            if len(files)>0:
                for crop in files:
                    if crop.endswith(image_ext):
                        crop_path = os.path.join(dirpath, crop)
                        class_type = str(crop).split(filename_splitter)[classname_idx]
                        class_type=class_type[0:str(class_type).index(image_ext)]
                        if class_type == "":
                            anonymous_classes.append(crop_path)
                            continue
                        class_num = class_counter[class_type]
                        if class_num <= num_per_class:
                            class_id=class_dictionary[class_type]
                            dest_file_path=os.path.join(mini_dataset_path,dirpath[len(src_dataset_path)+1:],crop)
                            shutil.copyfile(crop_path,dest_file_path)
                            print "copy %s to %s" %(crop_path,dest_file_path)
                            row = [dest_file_path, class_id]
                            class_num += 1
                            classes_of_mini_dataset.append(row)
                            class_counter[class_type] = class_num
                        else:
                            continue
        # np_classes_of_mini=np.array(classes_of_mini_dataset)
        # np.savetxt(csv_mini_dataset,classes_of_mini_dataset, delimiter=" ",fmt="%s %s")
        print "%d CLASSES COPIED." %len(classes_of_mini_dataset)
        return classes_of_mini_dataset

    def generate_mini_dataset_from_csv(self, src_dataset_csv="/a/b/classname:.csv", mini_dataset_csv="/a/b/c.csv",filename_splitter=" ",classname_idx=-1,num_per_class=0):
        class_dist=self.get_classname_distribution_from_csv(csv_path=src_dataset_csv,sep=filename_splitter)
        if num_per_class==0:
            num_per_class=class_dist[len(class_dist)-1][1] #get least number from counter
        classname_enum=list(enumerate([x for (x,y) in class_dist]))
        class_dictionary={classname:id for id,classname in classname_enum}
        class_counter = {classname: 1 for id, classname in classname_enum}
        anonymous_classes=[]
        df=pd.read_csv(src_dataset_csv,sep=filename_splitter)
        data_np=np.array(df)
        mini_dataset_rows=[]
        for row in data_np[:]:
            class_type=row[classname_idx]
            if class_type == "":
                anonymous_classes.append(row[0])
                continue
            class_num = class_counter[class_type]
            if class_num <= num_per_class:
                class_id = class_dictionary[class_type]
                mini_dataset_rows.append(row)
                class_num += 1
                class_counter[class_type] = class_num
            else:
                continue
        np_classes_of_mini=np.array(mini_dataset_rows)
        np.savetxt(mini_dataset_csv,np_classes_of_mini, delimiter=" ",fmt="%s")
        print "%d CLASSES COPIED." %len(np_classes_of_mini)
        return np_classes_of_mini

    def generate_dataset_hifreq_csv(self,hifreq_dataset="/a/b/c",src_csv="/a/b/c.csv",hifreq_csv="/a/b/c.csv",csv_sep=",",csv_class_idx=-1,filename_sep="_",filename_class_idx=-1,min_occur=50):
        counter=self.get_classname_distribution(dataset_path=hifreq_dataset)
        hifreq_classes = [x for (x, y) in counter if y >= min_occur]
        df=pd.read_csv(src_csv,sep=csv_sep)
        np_data=np.array(df)
        filtered_hifreq_rows=filter(lambda r:r[csv_class_idx] in hifreq_classes,np_data)
        np.savetxt(hifreq_csv,filtered_hifreq_rows,delimiter=csv_sep,fmt="%s")
        print "saved %d data to %s"%(len(filtered_hifreq_rows),hifreq_csv)
        return filtered_hifreq_rows



    def delete_files_in_folder(self,folder_path="/a/b/c",file_extension=".bin.png"):
        c = 0
        for dirpath,dirs,files in os.walk(folder_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(file_extension):
                        os.remove(os.path.join(dirpath,file))
                        print "REMOVED : %s" %file
                        c+=1
        print "TOTAL REMOVAL:%d" %c
        return

    def delete_content_of_text(self,text_path="/a/b/c.csv"):
        with open(text_path,"w") as file:
            file.truncate()
            file.close()
        return


    def split_train_test(self,dataset_txt_path="a.csv",train_txt_path="a.csv",test_txt_path="a.csv"):
        df=pd.read_csv(dataset_txt_path,sep="\s+",header=None,engine="python")
        data=np.array(df.ix[:,:])
        image_paths=np.array(data[:,0])
        class_labels=np.array(data[:,1])
        trains,tests=train_test_split(data,test_size=0.2)
        np.savetxt(train_txt_path,trains,delimiter=" ",fmt="%s %s")
        np.savetxt(test_txt_path, tests, delimiter=" ", fmt="%s %s")
        print "CREATED %d TRAININGs, %d TESTs" %(len(trains),len(tests))
        return

    def split_train_val_test(self,dataset_txt_path="a.csv",train_txt_path="a.csv",val_txt_path="a.csv",test_txt_path="a.csv"):
        df=pd.read_csv(dataset_txt_path,sep=" ",header=None,engine="python")
        data=np.array(df.ix[:,:])
        image_paths=np.array(data[:,0])
        class_labels=np.array(data[:,1])
        trains,vals=train_test_split(data,test_size=0.2)
        trains, tests = train_test_split(trains, test_size=len(vals))
        np.savetxt(train_txt_path,trains,delimiter=" ",fmt="%s %s")
        np.savetxt(val_txt_path, vals, delimiter=" ", fmt="%s %s")
        np.savetxt(test_txt_path, tests, delimiter=" ",fmt="%s %s")
        print "CREATED %d TRAININGs, %d VALS, %d TESTs" %(len(trains),len(vals),len(tests))
        return

    def convert_class_label_to_number(self, dataset_txt,class_dict):
        try:
            df = pd.read_csv(dataset_txt, sep=" ", header=None)
            data = np.array(df.ix[:, :])
            for i in range(data.shape[0]):
                class_type = data[i, -1]
                data[i, -1] = class_dict[class_type]
            np.savetxt(dataset_txt, data, delimiter=" ", fmt="%s %s")
            print "CONVERTED %d classes label to numbers" % data.shape[0]
        except Exception,e:
            print str(e)
        return

    def resize_image(self,image, resize_to, method=Image.ANTIALIAS,pad_color="black",src_avg_img_dim=[1776,2546]):
        image.thumbnail(resize_to, method)
        offset = (int((resize_to[0] - image.size[0]) / 2), int((resize_to[1] - image.size[1]) / 2))
        page = Image.new("RGB", resize_to, pad_color)  # RGB image process
        # page = Image.new("crossval1.old", resize_to, "white")	# Black-white image process
        page.paste(image, offset)

        return page

    def resize_all_zones(self,src_dataset,dest_dataset,resize_dim=[224,224],img_ext=".bin.png",pad_color="black"):
        i = 0
        misformed=[]
        self.imitate_image_filetree(src_dataset,dest_dataset)
        for dirpath,dirs,files in os.walk(src_dataset):
            if len(files)>0:
                for file in files:#sorted(glob.glob(os.path.join(dirpath, "*.*"))):
                    if str(file).endswith(img_ext):
                        file_path=os.path.join(dirpath,file)
                        image = Image.open(file_path).convert("RGB")  # RGB image process
                        # image = Image.open(file).convert("crossval1.old") # Black-white image process
                        resize_to = resize_dim  # height, width
                        new_size = self.resize_image(image, resize_to, pad_color=pad_color, method=Image.ANTIALIAS)
                        child_path=file_path[len(src_dataset)+1:]
                        resized_path=os.path.join(dest_dataset,child_path)
                        new_size.save(resized_path)#os.path.join(dest_dataset, basename(file)), ".png")
                        print "RESHAPED to: %s" %resized_path
                        i = i + 1
                    else:
                        misformed.append(file)
        print "TOTAL RESIZE: %d" %i
        print "MISFORMED: %s" % misformed

    def plot_class_distribution_from_csv(self,dataset_csv="a/b/c.csv",dct={}):
        df = pd.read_csv(dataset_csv, sep="\s+", delimiter="\s+", header=None, engine="python")
        data=np.array(df.values[:,0])
        y=np.array(df.values[:,-1])
        c = Counter(y)
        xlabels = [dct[k] for k, v in sorted(c.items())]
        classes=[k for k,v in sorted(c.items())]
        counts=[c[i] for i in classes]
        plt.bar(xlabels, counts)
        plt.xticks(rotation=45)
        plt.show()
        return classes,counts

    def convert_dataset_images_to_csv(self,image_dataset="a/b/c",image_extension=".bin.png",csv_dataset="d/e/f.csv",sep="_",class_idx=-1):
        counter=self.get_classname_distribution(dataset_path=image_dataset)
        class_dictionary = {key: idx for idx, key in enumerate([k for k, c in counter])}
        rows=[]
        if os.path.exists(image_dataset):
            for dirpath, dirs, files in os.walk(image_dataset):
                if len(files) > 0:
                    for file in files:
                        if str(file).endswith(image_extension):
                            class_label=str(file).split(sep)[class_idx]
                            class_label=class_label[0:str(class_label).index(image_extension)]#.replace(image_extension,"")
                            if class_label !="":
                                fp = os.path.join(dirpath, file)
                                #fn=class_label.replace(image_extension,"")
                                class_id=class_dictionary[class_label]
                                rows.append([fp,class_id])
            np_data=np.array(rows)
            np.savetxt(csv_dataset, np_data, delimiter=" ", fmt="%s %s")
            print "CONVERTED %d paths to %s"%(np_data.shape[0],csv_dataset)
        else:
            raise Exception("PATH %s NOT EXISTS" %image_dataset)
        return class_dictionary

    def get_class_dictionary(self):
        class_dict = {
            'paragraph': 1,
            'page-number': 2,
            'catch-word': 3,
            'header': 4,
            'heading': 5,
            'signature-mark': 6,
            'other': 7,
            'Separator': 8,
            'footnote': 9,
            'marginalia': 10,
            'Graphic': 11,
            'Maths': 12,
            'caption': 13,
            'Table': 14,
            'footnote-continued': 15,
            'endnote': 16
        }
        return class_dict

    # def split_train_val_test(self,X=np.array([]),y=np.array([])):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=crossval1.old)
    #     X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=int(0.2*len(X)), random_state=crossval1.old)
    #     return X_train,X_val,X_test,y_train,y_val,y_test

    def map_coordinate_to_input_system(self,coord=[0,0],img_dim=[5,5],src_dim=[500,500],dest_dim=[224,224]):
        x0=coord[0]
        y0=coord[1]
        w0=img_dim[0]
        h0=img_dim[1]
        src_dim_width=src_dim[0]
        src_dim_height = src_dim[1]
        dest_dim_width = dest_dim[0]
        dest_dim_height = dest_dim[1]
        #predict the mapped dimension
        w1=0
        h1=0
        if w0>h0:
            w1=dest_dim_width
            h1=(h0/w0)*w1
        else:
            h1=dest_dim_height
            w1=(w0/h0)*h1
        #predict the map
        x1=(x0/src_dim_width)*dest_dim_width
        y1 = (y0 / src_dim_height) * dest_dim_height
        #check if oversize
        x1_1=x1+(w1-1)
        y1_1=y1+(h1-1)
        if x1_1>=dest_dim_width:
            diff=x1_1-dest_dim_width
            x1-=diff
        if y1_1>=dest_dim_height:
            diff=y1_1-dest_dim_height
            y1-=diff
        return int(x1),int(y1),int(w1),int(h1)

    # def get_avg_dim_of_dataset_images(self,GT_path="a/b/c",GT_ext=".xml"):
    #     widths=[]
    #     heights=[]
    #     for dirpath,dirs,files in os.walk(GT_path):
    #         if len(files)>0:
    #             for file in files:
    #                 if str(file).endswith(GT_ext):
    #                     if "jpg" in GT_ext:
    #                        img=Image.open(os.path.join(dirpath,file))
    #                        w,h=img.size
    #                        widths.append(w)
    #                        heights.append(h)
    #                        print "append w:%d, h:%d" % (w, h)
    #                     if "xml" in GT_ext:
    #                         try:
    #                             xml_file = os.path.join(dirpath, file)
    #                             tree = ET.parse(xml_file)
    #                             root = tree.getroot()
    #                             img_f = ""
    #                             main_children = root.getchildren()
    #                             for x in root:
    #                                 # print x.tag
    #                                 if str(x.tag).endswith("Page"):
    #                                     w=int(x.attrib["imageWidth"])
    #                                     h=int(x.attrib["imageHeight"])
    #                                     widths.append(w)
    #                                     heights.append(h)
    #                                     print "append w:%d, h:%d" %(w,h)
    #                         except Exception, e:
    #                             print "Error:%s" % str(e)
    #                             continue
    #     avg_width=int(math.floor(np.mean(widths)))
    #     avg_height=int(math.floor(np.mean(heights)))
    #     print "AVG W:%d, h:%d" %(avg_width,avg_height)
    #     return avg_width,avg_height

    def get_dataset_average_image_dimension(self,dataset_path="/a/b/c",img_ext=".bin.png"):
        img_widths=[]
        img_heights=[]
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        img_path=os.path.join(dirpath,file)
                        try:
                            with Image.open(img_path) as img:
                                w, h = img.size
                                img_widths.append(w)
                                img_heights.append(h)
                                img.load()
                        except Exception,e:
                            print str(e)
                            continue
        avg_width=int(np.mean(img_widths))
        avg_height = int(np.mean(img_heights))
        return avg_width,avg_height


    def copy_files(self,src_path="a/b/c",dest_path="d/e/f",file_ext=".xml",keep_tree=True):
        if keep_tree:
            self.imitate_filetree(src_path,dest_path)
        c=0
        for dirpath,dirs,files in os.walk(src_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(file_ext):
                        src_xml_path=os.path.join(dirpath,file)
                        if keep_tree:
                            dest_xml_path=src_xml_path[len(src_path)+1:]
                            dest_xml_path=os.path.join(dest_path,dest_xml_path)
                        else:
                            dest_xml_path =os.path.join(dest_path,file)
                        shutil.copyfile(src_xml_path, dest_xml_path)
                        print "COPIED FILE : %s" % dest_xml_path
                        c+=1
        print "%d FILES COPIED" %c

    def rename_images(self,GT_folder="a/b/c",GT_ext=".xml",image_dataset="a/b/c"):
        count=0
        bad_xml=[]
        for dirpath,dirs,files in os.walk(GT_folder):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(GT_ext):
                        xml_path=os.path.join(dirpath,file)
                        try:
                            zones,mistagged=self.extract_zones_from_xml(xml_path)
                            for zone in zones:
                                filename = (zone[0]).split("/")[-1] #'src/jpg/image.jpg.xml'
                                filename = filename.replace(".jpg", "")
                                temp_dir=dirpath.replace("page","jpg") #'src/a/b/c/page -> src/a/b/c/'
                                temp_dir=temp_dir[len(GT_folder):]
                                dest_path=os.path.join(image_dataset,temp_dir)
                                zone_type = zone[1]
                                zone_coord = zone[2]
                                c = []
                                for z in zone_coord:
                                    xy = z.split(",")
                                    x=int(xy[0])
                                    y=int(xy[1])
                                    z = [x, y]  #tuple of coord [[0,0],[2,2],]
                                    c.append(z)
                                    break #get only first coord
                                x=c[0][0]
                                y=c[0][1]
                                page_w=zone[3][0]
                                page_h = zone[3][1]
                                filename=filename+"_"+zone_type+".bin.png"
                                newname=str(page_w)+"_"+str(page_h)+"_"+str(x)+"_"+str(y)+"_"+filename
                                old_path=os.path.join(dest_path,filename)
                                new_path=os.path.join(dest_path,newname)
                                if not os.path.exists(new_path):
                                    if os.path.exists(old_path):
                                        os.rename(old_path,new_path)
                                        print "RENAMED to %s"%new_path
                                        count+=1


                        except Exception, e:
                            print "Error:%s" % str(e)
                            bad_xml.append(xml_path)
                            continue
        print "%d FILES RENAMED"%count
        print "BAD XMLS:\n"
        for x in bad_xml:
            print "%s"%x

    def reshape_and_relocate_zones(self,src_dataset,
                                   dest_dataset,
                                   resize_dim=[227,227],
                                   img_ext=".bin.png",
                                   pad_color="white",
                                   filename_sep="_",
                                   classname_idx=-1,
                                   relocate=True,
                                   average_page_dim=[1776,2546]):
        aug=[]
        i = 0
        misformed=[]
        error_files = []
        self.imitate_filetree(src_dataset,dest_dataset)
        reshaped_zones=[]
        for dirpath,dirs,files in os.walk(src_dataset):
            if len(files)>0:
                for file in files:#sorted(glob.glob(os.path.join(dirpath, "*.*"))):
                    if str(file).endswith(img_ext):
                        try:
                            splitted = str(file).split(filename_sep)
                            page_w = int(splitted[0])
                            page_h = int(splitted[1])
                            x = int(splitted[2])
                            y = int(splitted[3])
                            if page_w == 0:
                                page_w = average_page_dim[0]
                            if page_h == 0:
                                page_h = average_page_dim[1]  # average height of images in dataset
                            file_path = os.path.join(dirpath, file)
                            child_path = file_path[len(src_dataset) + 1:]
                            resized_path = os.path.join(dest_dataset, child_path)
                            image = Image.open(file_path).convert("RGB")  # RGB image process
                            if relocate:
                                zone_w, zone_h = image.size
                                # image = Image.open(file).convert("crossval1.old") # Black-white image process
                                resize_to = resize_dim  # height, width

                                x1, y1, w1, h1 = self.map_coordinate_to_input_system(coord=[x, y],
                                                                                     img_dim=[zone_w, zone_h],
                                                                                     src_dim=[page_w, page_h],
                                                                                     dest_dim=resize_dim)
                                image.thumbnail(resize_to, Image.ANTIALIAS)
                                # offset = (int((resize_to[0] - image.size[0]) / 2), int((resize_to[crossval1.old] - image.size[crossval1.old]) / 2))
                                page = Image.new("RGB", resize_to, pad_color)  # RGB image process
                                # page = Image.new("crossval1.old", resize_to, "white")	# Black-white image process
                                page.paste(image, (x1, y1))
                                page.save(resized_path)  # os.path.join(dest_dataset, basename(file)), ".png")
                                aug.append(resized_path)
                                image.load()
                                page.load()
                            else:
                                resized_image = image.resize((resize_dim[0], resize_dim[1]))
                                resized_image.save(resized_path)
                                aug.append(resized_path)
                            print "RESHAPED to: %s" % resized_path
                            i = i + 1
                        except Exception,e:
                            #print "ERROR:%s\nFILE:%s" %(str(e),os.path.join(dirpath,file))
                            error_files.append(os.path.join(dirpath,file))

                    else:
                        misformed.append(file)
        print "TOTAL RESIZE: %d" %i
        print "MISFORMED: %s" % misformed
        print "Error Files:: %s" % error_files
        return aug

    def reshape_relocate_zones_then_merge_to_images(self,src_dataset,
                                   dest_dataset,
                                   resize_dim=[227,227],
                                   img_ext=".bin.png",
                                   pad_color="white",
                                   filename_sep="_",
                                   classname_idx=-1,
                                   relocate=True,
                                   average_page_dim=[1776,2546]):
        aug=[]
        i = 0
        misformed=[]
        error_files = []
        self.imitate_filetree(src_dataset,dest_dataset)
        tempdir="/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/temp2"
        tempdir2 = "/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/temp3"
        for dirpath,dirs,files in os.walk(src_dataset):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        try:
                            splitted = str(file).split(filename_sep)
                            page_w = int(splitted[0])
                            page_h = int(splitted[1])
                            x = int(splitted[2])
                            y = int(splitted[3])
                            if page_w == 0:
                                page_w = average_page_dim[0]
                            if page_h == 0:
                                page_h = average_page_dim[1]  # average height of images in dataset
                            file_path = os.path.join(dirpath, file)
                            child_path = file_path[len(src_dataset) + 1:]
                            resized_path = os.path.join(dest_dataset, child_path)
                            image = Image.open(file_path).convert("RGB")  # RGB image process
                            if relocate:
                                zone_w, zone_h = image.size
                                # image = Image.open(file).convert("crossval1.old") # Black-white image process
                                resize_to = resize_dim  # height, width

                                x1, y1, w1, h1 = self.map_coordinate_to_input_system(coord=[x, y],
                                                                                     img_dim=[zone_w, zone_h],
                                                                                     src_dim=[page_w, page_h],
                                                                                     dest_dim=resize_dim)
                                image.thumbnail(resize_to, Image.ANTIALIAS)
                                # offset = (int((resize_to[0] - image.size[0]) / 2), int((resize_to[crossval1.old] - image.size[crossval1.old]) / 2))

                                page = Image.new("RGB", resize_to, "white")  # RGB image process
                                page.paste(image, (x1, y1))
                                tempfp=os.path.join(tempdir,file)
                                page.save(tempfp)
                                self.binarize_img_with_otsu(tempfp,tempfp)

                                page2 = Image.new("RGB", resize_to, "black")  # RGB image process
                                page2.paste(image, (x1, y1))
                                tempfp2 = os.path.join(tempdir2, file)
                                page2.save(tempfp2)
                                self.binarize_img_with_otsu(tempfp2, tempfp2)

                                self.resize_image(file_path, resized_path, resize_dim=resize_dim)
                                self.binarize_img_with_otsu(resized_path,resized_path)

                                self.merge_reloc_and_unreloc_image(resized_path,tempfp,tempfp2,resized_path)
                                os.remove(tempfp)
                                os.remove(tempfp2)
                                aug.append(resized_path)
                                image.load()
                                page.load()
                                page2.load()
                            else:
                                resized_image = image.resize((resize_dim[0], resize_dim[1]))
                                resized_image.save(resized_path)
                                aug.append(resized_path)
                            print "RESHAPED to: %s" % resized_path
                            i = i + 1
                        except Exception,e:
                            #print "ERROR:%s\nFILE:%s" %(str(e),os.path.join(dirpath,file))
                            error_files.append(os.path.join(dirpath,file))

                    else:
                        misformed.append(file)
        print "TOTAL RESIZE: %d" %i
        print "MISFORMED: %s" % misformed
        print "Error Files:: %s" % error_files
        return aug


    # def select_hifreq_class(self,class_to_omit=["a","b"],dataset_folder="/a/b/c",img_ext=".TIF",classname_splitter="_"):
    #     rmv=0
    #     p=re.compile("[a-zA-Z\-_.]+")
    #     for dirpath,dirs,files in os.walk(dataset_folder):
    #         if len(files)>0:
    #             for file in files:
    #                 f=file[0:str(file).index(img_ext)]
    #                 minor_class_key=[s for s in f.split(classname_splitter) if s in class_to_omit]
    #                 if len(minor_class_key)>0:
    #                     fp=os.path.join(dirpath,file)
    #                     os.remove(fp)
    #                     rmv+=crossval1.old
    #                     print "removed %s"%fp
    #     print "%d files removed" %rmv


    def generate_highfreq_class_dataset(self, src_dataset_path="a/b/c",dest_dataset_path="a/b/c",img_ext=".bin.png",sep="_",classname_idx=-1,min_occur=50):
        counter=self.get_classname_distribution(dataset_path=src_dataset_path)
        hifreq_class=[x for (x,y) in counter if y>=min_occur]
        low_freq=[]
        high_freq = []
        self.imitate_filetree(src_dataset_path,dest_dataset_path)
        for dirpath,dirs,files in os.walk(src_dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        spli=str(file).split(sep)
                        class_name=spli[classname_idx]
                        class_name=class_name.replace(img_ext,"")
                        if class_name in hifreq_class:
                            src_fp=os.path.join(dirpath,file)
                            dst_fp=dirpath[len(src_dataset_path)+1:]
                            dst_fp=os.path.join(dest_dataset_path,dst_fp,file)
                            shutil.copyfile(src_fp,dst_fp)
                            print "copied %s" %file
                            high_freq.append(class_name)
                        else:
                            low_freq.append(class_name)
                            continue
        len_hifreq=len(high_freq)
        set_hifreq=set(high_freq)
        len_lofreq=len(low_freq)
        set_lofreq=set(low_freq)
        print "hi freq classes number: %d" %len_hifreq
        print "high freq classes name:%d\n%s" % (len(set_hifreq),set_hifreq)
        print "low freq classes number: %d" % (len_lofreq)
        print "low freq classes:%d\n%s"%(len(set_lofreq),set_lofreq)

    def standardize_naming(self,src_img_path="/a/b/c",dataset_path="/a/b/c",img_ext=".TIF"):
        renamed=0
        p=re.compile("[0-9]+_")
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        parent_dir=str(dirpath).split("/")[-1]
                        page_image_name=parent_dir+"BIN.TIF"
                        page_image_path=os.path.join(src_img_path,page_image_name)
                        page_image=Image.open(page_image_path)
                        keep=page_image.copy()
                        w,h=keep.size
                        page_image.close()
                        nums=p.findall(file)
                        fp=os.path.join(dirpath,file)
                        newname=str(file).replace("text_","").replace("_non-text","").replace("text-with-special-symbols_","")
                        spl = newname.split("_")
                        new = str(w)+"_"+str(h)+"_" + spl[0] + "_" + spl[1] + "_" + spl[-1]
                        newfp=os.path.join(dirpath,new)
                        os.rename(fp,newfp)
                        renamed+=1
                        print "renamed %s to %s" %(file,newname)

        print "%d files renamed." %renamed

    def generate_no_redundancy_dataset(self,src="/a/b/c",dest="/a/b/c"):
        no_redundant_folder = ['J03C', 'D055', 'E003', 'A002', 'E04C', 'C03P', 'D059', 'IG0B', 'H015', 'J008', 'E019',
                               'E009', 'I03D', 'A00B', 'K00A', 'H00T', 'E046', 'J04L', 'A037', 'E038', 'D05C', 'A06F',
                               'C00G', 'E047', 'D03H', 'A056', 'E034', 'C048', 'I04J', 'A00E', 'H00C', 'H03F', 'E037',
                               'A055', 'J04N', 'J03D', 'D04N', 'J04R', 'A00O', 'A06J', 'D06C', 'A001', 'K009', 'E02I',
                               'D03E', 'E03J', 'C00I', 'IG0F', 'C04K', 'D03P', 'E03L', 'C04C', 'C032', 'C00J', 'E03D',
                               'E045', 'E010', 'H03J', 'H008', 'A05J', 'E04F', 'IG00', 'J04S', 'I04L', 'A06H', 'H04B',
                               'E02A', 'A042', 'H007', 'A05Q', 'I03K', 'H00X', 'I03J', 'C00E', 'A064', 'J04B', 'E04L',
                               'E00P', 'H04G', 'D03D', 'D065', 'H00B', 'J002', 'C000', 'H00W', 'H04D', 'J03K', 'E01D',
                               'IG0I', 'K003', 'IG0C', 'C041', 'C03L', 'I04G', 'C039', 'E03M', 'D05Q', 'C006', 'E00E',
                               'C00C', 'E03P', 'V000', 'A058', 'IG0E', 'A00D', 'V00K', 'H00G', 'V00B', 'J034', 'H04Q',
                               'A004', 'A03I', 'E00D', 'I03B', 'H00K', 'H00O', 'K00K', 'V00H', 'C049', 'H047', 'D06B',
                               'E03F', 'J00A', 'J037', 'H00P', 'J009', 'E012', 'H045', 'E00N', 'J00P', 'J03P', 'A03M',
                               'E040', 'H04F', 'D05E', 'A06N', 'D06A', 'I04C', 'D06F', 'IG0M', 'D05D', 'H002', 'C005',
                               'I04K', 'H00F', 'E04D', 'I03I', 'D06R', 'E02K', 'D031', 'E03E', 'H005', 'IG05', 'E044',
                               'D03L', 'A051', 'A06K', 'E04I', 'J03Q', 'V00F', 'E00F', 'K00F', 'J03F', 'D06N', 'C04P',
                               'A06B', 'A04N', 'E029', 'E03N', 'C031', 'J04I', 'I03E', 'J04G', 'E02H', 'D06Q', 'V006',
                               'H04N', 'J00Q', 'A005', 'D057', 'E04E', 'C040', 'A040', 'C035', 'E01A', 'H030', 'E01J',
                               'A05H', 'K00B', 'A06C', 'IG01', 'H00E', 'A03E', 'D04K', 'D03N', 'D05M', 'J043', 'E02M',
                               'A05F', 'A054', 'H03E', 'E018', 'D054', 'H037', 'E021', 'D06J', 'C038', 'C04F', 'A05M',
                               'J038', 'E005', 'I03C', 'A039', 'A04I', 'H034', 'I04H', 'J03B', 'K001', 'A04G', 'A04C',
                               'V00M', 'D05F', 'H04A', 'A036', 'IG0D', 'D038', 'I03N', 'E039', 'A05C', 'C001', 'K00J',
                               'A03H', 'H03N', 'E01I', 'A04K', 'D04C', 'E04M', 'E049', 'D063', 'A03N', 'A03G', 'I03P',
                               'D05I', 'J03G', 'E03S', 'J00F', 'A04E', 'D05B', 'C04A', 'D06S', 'H00L', 'H041', 'A06I',
                               'H04P', 'V001', 'E01E', 'D05K', 'H00V', 'A041', 'J00E', 'J045', 'A05P', 'J04E', 'C04H',
                               'H04K', 'V002', 'E00H', 'E004', 'C03F', 'E03T', 'C043', 'D035', 'A03B', 'C037', 'E02F',
                               'J006', 'D06L', 'A00L', 'J00M', 'J00D', 'H04M', 'E02E', 'A00I', 'H00A', 'K00C', 'J032',
                               'E02N', 'H006', 'A00H', 'J00N', 'E04B', 'H03G', 'E041', 'D040', 'J04A', 'K00M', 'E020',
                               'D066', 'J033', 'A069', 'C04B', 'E03K', 'A007', 'E017', 'C04N', 'A035', 'H00Z', 'E02B',
                               'I04B', 'D039', 'E000', 'E011', 'D032', 'A05N', 'J005', 'H00U', 'C04D', 'A068', 'V00I',
                               'V004', 'V00N', 'H01B', 'A06A', 'C04E', 'H04H', 'J003', 'A062', 'D03C', 'H018', 'D04J',
                               'H03P', 'J048', 'C047', 'J04P', 'E036', 'D033', 'C00F', 'C00K', 'D050', 'E024', 'I04E',
                               'I04I', 'H013', 'A05L', 'J004', 'E008', 'H004', 'E03H', 'E03B', 'IG03', 'V00C', 'C04I',
                               'H00H', 'A050', 'H043', 'E031', 'E033', 'J04C', 'C03E', 'V005', 'E03I', 'A04M', 'D037',
                               'A031', 'D045', 'H010', 'D069', 'A00K', 'E02J', 'H00J', 'A03F', 'A00C', 'D052', 'K00H',
                               'V009', 'IG0K', 'D058', 'E04P', 'E043', 'A06L', 'H01A', 'K00E', 'H003', 'A006', 'V007',
                               'J00H', 'D04B', 'A046', 'C00D', 'C036', 'E03A', 'D05L', 'C03M', 'D04M', 'J04J', 'A06M',
                               'A05K', 'J03E', 'A034', 'J03J', 'D04I', 'E030', 'H001', 'I04D', 'H017', 'A03C', 'D064',
                               'E01K', 'D06G', 'H00Y', 'H00Q', 'H03C', 'A03L', 'E028', 'A00M', 'V003', 'D053', 'A03J',
                               'D03K', 'E04K', 'I03A', 'J042', 'H011', 'A065', 'K006', 'I04M', 'D06K', 'A038', 'D056',
                               'E04H', 'J00O', 'IG0A', 'H04C', 'E00I', 'D03G', 'A044', 'H00D', 'C00H', 'C03C', 'D05G',
                               'H01D', 'E02C', 'I04F', 'C03I', 'A06G', 'D030', 'A05A', 'E01L', 'A03D', 'D041', 'D05J',
                               'K00O', 'A032', 'E00A', 'J03L', 'C044', 'D04E', 'A030', 'C007', 'V00L', 'A05D', 'E023',
                               'E01G', 'J00L', 'C042', 'H03A', 'E016', 'J007', 'J04H', 'E014', 'E035', 'E01B', 'D04D',
                               'I03H', 'D03J', 'A06D', 'V00G', 'A04A', 'H014', 'D036', 'C03H', 'H00R', 'H044', 'K00N',
                               'A061', 'I03G', 'C03N', 'D048', 'H031', 'D046', 'D06H', 'J000', 'E00L', 'K00P', 'J00K',
                               'C00A', 'A003', 'H042', 'I03M', 'A033', 'D049', 'D05A', 'H032', 'IG0G', 'D03F', 'E002',
                               'I04N', 'H046', 'I03L', 'C008', 'D047', 'D062', 'H03K', 'C046', 'J03N', 'J04K', 'E04N',
                               'H03I', 'E00M', 'A052', 'A03P', 'A00P', 'H00I', 'D05H', 'H04E', 'K00L', 'J00J', 'J035',
                               'D068', 'H000', 'V00E', 'E00B', 'E01F', 'A04B', 'E02L', 'J03A', 'H03B', 'C009', 'IG02',
                               'E02D', 'E00Q', 'J04F', 'E01M', 'E015', 'V00D', 'C003', 'H03H', 'E01C', 'A04H', 'H00M',
                               'H033', 'E032', 'A00J', 'H04I', 'C03K', 'E022', 'C00B', 'D067', 'I04A', 'J04D', 'A04L',
                               'D044', 'H01C', 'V008', 'H03M', 'H00N', 'V00A', 'D03I', 'J00I', 'D05N', 'J044', 'A045',
                               'C03O', 'H049', 'IG0J', 'D04A', 'J040', 'H03D', 'C04J', 'J00C', 'E026', 'E03Q', 'D043',
                               'E01H', 'A00A', 'H012', 'A063', 'K00I', 'C045', 'C03B', 'A053', 'C03D', 'H036', 'K008',
                               'J03I', 'H019', 'J046', 'A008', 'E042', 'D060', 'A048', 'E027', 'J031', 'J03H', 'D04P',
                               'J001', 'C04G', 'D05P', 'H048', 'J00B', 'A043', 'K007', 'D04L', 'E00G', 'H04J', 'V00J',
                               'J00G', 'IG09', 'K002', 'H040', 'J036', 'A03A', 'C04L', 'D06P', 'E02G', 'E00C', 'C033',
                               'A057', 'J049', 'D051', 'A049', 'H03L', 'A04P', 'E00K', 'E048', 'IG0H', 'D04H', 'IG04',
                               'IG0L', 'A00G', 'A047', 'I03F', 'H00S', 'E001', 'C034', 'C04M', 'C03G', 'J041', 'D06M',
                               'E04G', 'D04G', 'E03R', 'C002', 'D03M', 'E013', 'A03K', 'E00J', 'A066', 'J03M', 'H039',
                               'A00F', 'A059', 'J04Q', 'A05B', 'A04F', 'J030', 'H04L', 'C004', 'D06I', 'IG06', 'E04J',
                               'J04M', 'H009', 'D042', 'E03G', 'H016', 'E025', 'A067', 'E04A', 'D034', 'A009', 'A060',
                               'H035', 'E007', 'C03A', 'A05E', 'A06E', 'J047', 'D04F', 'H038', 'K004', 'D061', 'C030']
        for folder in no_redundant_folder:
            srcp = os.path.join(src, folder)
            destp = os.path.join(dest, folder)
            copytree(srcp, destp)
            print "copied %s" % folder

    def removed_file_by_regex(self,src_img_path="/a/b/c", img_ext=".bin.png",regex="^[0-9_]+"):
        removed = 0
        remained = 0
        p = re.compile(regex)
        for dirpath, dirs, files in os.walk(src_img_path):
            if len(files) > 0:
                for file in files:
                    if str(file).endswith(img_ext):
                        begin = p.findall(file)
                        if len(begin) == 0:
                            fp = os.path.join(dirpath, file)
                            os.remove(fp)
                            print "removed %s" % fp
                            removed += 1
                        else:
                            remained += 1
        print "%d files removed." % removed
        print "%d files remained." % remained
        return

    def get_classname_distribution_from_csv(self,csv_path="/a/b/c.csv",sep=" ",
                                            class_id_is_label=False):
        df=pd.read_csv(filepath_or_buffer=csv_path,sep=sep,header=None)
        datas=np.array(df)
        X=datas[:,:-1]
        y=datas[:,-1]
        classn=[]
        if class_id_is_label:
            classn=y
        else:
            for x in X:
                cn=str(x).split("_")[-1]
                cn=str(cn).replace(".bin.png","")
                classn.append(cn)
        counter=Counter(classn)
        #counter=Counter(y)
        most_common=counter.most_common()
        return most_common

    def get_difference_two_folders(self,dir1="/a/b/c",dir2="/a/b/c",file_ext=".bin.png"):
        list1=[]
        list2=[]
        for dirpath,dirs,files in os.walk(dir1):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(file_ext):
                        list1.append(file)
        for dirpath,dirs,files in os.walk(dir2):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(file_ext):
                        list2.append(file)
        set1=set(list1)
        set2 = set(list2)
        diff=[]
        if len(set1)>len(set2):
            diff=[x for x in set1 if x not in set2]
        else:
            diff = [x for x in set2 if x not in set1]
        return len(set1),len(set2),diff


    def get_classname_dictionary(self,dataset_path,img_ext=".bin.png",filename_sep="_",classname_idx=-1):
        class_dist=self.get_classname_distribution(dataset_path=dataset_path,img_ext=img_ext,name_splitter=filename_sep,start_classname_idx=classname_idx)
        classname_enum = list(enumerate([x for (x, y) in class_dist]))
        class_dictionary = {id:classname for id, classname in classname_enum}
        return class_dictionary

    def get_classname_dictionary_from_csv(self,dataset_csv,img_ext=".bin.png",sep=" ",classname_idx=-1):
        # class_dist=self.get_classname_distribution_from_csv(dataset_csv,sep)
        # classname_enum = list(enumerate([x for (x, y) in class_dist]))
        # class_dictionary = {x: y for x, y in classname_enum}
        df=pd.read_csv(dataset_csv,sep=sep,header=None)
        npd=np.array(df)
        pairs=[]
        for row in npd:
            x=row[0]
            y=row[classname_idx]
            clasname=str(x).split("_")[-1].replace(img_ext,"")
            pairs.append([y,  clasname])
        pairs=[list(x) for x in set(tuple(x) for x in pairs)]
        class_dictionary = {x:y for x, y in pairs}

        return class_dictionary




    def convert_dataset_to_csv(self,dataset_path="/a/b/c",
                               csv_path="/a/b/c.csv",
                               img_ext=".bin.png",
                               csv_sep=" ",
                               filename_sep="_",
                               filename_class_idx=-1):
        if not os.path.exists(csv_path):
            with open(csv_path,"w"): pass
        class_dictionary=self.get_classname_dictionary(dataset_path=dataset_path,img_ext=img_ext,filename_sep=filename_sep,classname_idx=filename_class_idx)
        class_dictionary={v:k for k,v in class_dictionary.items()}
        rows=[]
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        split=str(file).split(filename_sep)
                        classname=split[filename_class_idx]
                        classname=classname[0:str(classname).index(img_ext)]
                        classname_id=class_dictionary[classname]
                        img_path=os.path.join(dirpath,file)
                        row=[img_path,classname_id]
                        rows.append(row)
        rows_np=np.array(rows)
        np.savetxt(csv_path,rows_np,delimiter=csv_sep,fmt="%s")
        print "saved %d data to %s"%(len(rows_np),csv_path)
        return class_dictionary

    def get_pixel_distribution_from_image_path(self,img_path="/a/b/c.png",channel_idx=0):
        image = cv2.imread(img_path)
        h, w, channels = image.shape
        # img=Image.open(img_path)
        # img_rgb=img.convert("RGB")
        # w,h=img.size
        pixs=[]
        for y in range(0,h-1):
            for x in range(0,w-1):
                # r,g,b=img_rgb.getpixel((x,y))
                # pixs.append(r)
                # pixs.append
                try:
                    pixs.append(image[y, x,channel_idx])
                except Exception,e:
                    #print "error:%s"%str(e)
                    continue
        counter=Counter(pixs)
        counter=counter.most_common()
        return counter

    # def get_pixel_distribution_from_image(self,rgb_img_instance):
    #     img=rgb_img_instance
    #     w,h=img.size
    #     pixs=[]
    #     for y in range(0,h):
    #         for x in range(0,w):
    #             r,g,b=img.getpixel((x,y))
    #             pixs.append(r)
    #             pixs.append(g)
    #             pixs.append(b)
    #     counter=Counter(pixs)
    #     counter=counter.most_common()
    #     return counter

    def get_pixels_binarization_status_from_dataset_csv(self,csv_path="/a/b/c.csv",sep=" ",sample_num=40):
        df=pd.read_csv(filepath_or_buffer=csv_path,sep=sep,header=None)
        npy=np.array(df)
        paths=npy[:,0]
        part0,tests =train_test_split(paths,test_size=sample_num)
        gray_imgs=[]
        is_binarized=True
        for test in tests:
            c=self.get_pixel_distribution_from_image_path(test)
            if len(c)>2:#if not binary
                is_binarized=False
                gray_imgs.append(test)
        print is_binarized
        return is_binarized,gray_imgs

    def generate_hifreq_dataset_csv_from_csv(self, src_csv="/a/b/c.csv",dest_csv="/a/b/c.csv",src_csv_sep=",",src_csv_classname_idx=-1,dest_csv_sep=",",dest_csv_classname_idx=-1,min_occur=50):
        class_dist=self.get_classname_distribution_from_csv(csv_path=src_csv,sep=src_csv_sep)
        hifreq_class = [x for (x, y) in class_dist if y >= min_occur]
        df=pd.read_csv(src_csv,header=None)
        data=np.array(df)
        hifreq_data=filter(lambda x:x[src_csv_classname_idx] in hifreq_class,data)
        np.savetxt(dest_csv,hifreq_data,delimiter=",",fmt="%s")
        print "saved %d data to %s" %(len(hifreq_data),dest_csv)

    def get_image_channel_matrix(self, img_path="/a/b/c.png"):
        image = cv2.imread(img_path)
        rows,cols,channels=image.shape
        pixs_b=[]
        pixs_g = []
        pixs_r = []
        for x in range(rows):
            for y in range(cols):
                pixs_b.append(image[x, y,0])
                pixs_g.append(image[x, y, 1])
                pixs_r.append(image[x, y, 2])
        b=np.reshape(pixs_b,[227,227,1])
        g = np.reshape(pixs_g, [227, 227,1])
        r = np.reshape(pixs_r, [227, 227,1])
        merged = np.concatenate([b,g,r],axis=2)
        return pixs_b,pixs_g,pixs_r,merged


    def get_database_average_pixel(self,dataset_path="/a/b/c",img_ext=".jpg"):
        pix_val=[0,0,0]
        pix_count=0
        i=0
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        try:
                            filepath = os.path.join(dirpath, file)
                            image = cv2.imread(filepath)
                            rows, cols, channels = image.shape
                            pix_count += (rows * cols)
                            for x in range(rows):
                                for y in range(cols):
                                    pix_val[0] += image[x, y, 0]
                                    # pix_val[crossval1.old] += image[x, y, crossval1.old]
                                    # pix_val[2] += image[x, y, 2]
                                    print pix_val[0]/pix_count
                            #print "%d/400000"%i
                            i+=1
                        except Exception,e:
                            print "ERROR:%s"%file

        avg_pix=[p/pix_count for p in pix_val]#pix_val/pix_count
        return avg_pix

    def get_image_black_pix_ratio(self, img_path="/a/b/c.bin.png",channel=0):
        image = cv2.imread(img_path)
        rows,cols,channels=image.shape
        black_pix_count=0
        for x in range(rows):
            for y in range(cols):
                pix=image[x, y,channel]
                if pix==0:
                    black_pix_count+=1
        ratio=black_pix_count/(rows*cols)
        return ratio

    def remove_black_pixels_majored_images(self,dataset_path="/a/b/c",img_ext=".bin.png",channel=0,black_pecentage_threshold=0.95):
        black_pix_imgs = 0
        for dirpath, dirs, files in os.walk(dataset_path):
            if len(files) > 0:
                for file in files:
                    if str(file).endswith(img_ext):
                        filepath = os.path.join(dirpath, file)
                        black_ratio=self.get_image_black_pix_ratio(img_path=filepath,channel=channel)
                        if black_ratio > black_pecentage_threshold:
                            os.remove(filepath)
                            print "removed %s" %filepath
                            black_pix_imgs+=1
        print "%d images are removed" %black_pix_imgs
        return


    def remove_black_pixels_majored_images_from_csv(self,dataset_path_csv="/a/b/c.csv",non_blacked_dataset_path_csv="/a/b/c.csv",sep=" ",filepath_idx=0,class_type_idx=-1,img_ext=".bin.png",channel=0,black_pecentage_threshold=0.95):
        non_black_pix_majored_images_row=[]
        black_pix_majored_images_row = 0
        df=pd.read_csv(dataset_path_csv,sep=sep,header=None)
        np_data=np.array(df)
        for row in np_data[:]:
            filepath=row[filepath_idx]
            class_type=row[class_type_idx]
            black_ratio=self.get_image_black_pix_ratio(img_path=filepath)
            if black_ratio < black_pecentage_threshold:
                non_black_pix_majored_images_row.append(row)
                print "added class %s" %row[class_type_idx]
            else:
                black_pix_majored_images_row+=1
        np.savetxt(non_blacked_dataset_path_csv,non_black_pix_majored_images_row,delimiter=sep,fmt="%s")
        print "saved %d filepaths" %len(non_black_pix_majored_images_row)
        print "found %d black_pix_majored_images" %black_pix_majored_images_row
        return

    def generate_balanced_train_val_by_quantity(self,dataset_csv="/a/b/c.csv",
                                                train_csv="/a/b/c.csv",
                                                val_csv="/a/b/c.csv",
                                                test_csv="/a/b/c.csv",
                                                remain_csv=None,
                                                train_num=1200,
                                                val_num=100,
                                                test_num=100,
                                                sep=" ",
                                                img_ext=".bin.png",
                                                class_type_idx=-1,
                                                fold_num=10,
                                                test_exclusion_reg=["_ZOOM","_SHEAR","_TOPLEFTCROP"]):
        df=pd.read_csv(dataset_csv,sep=sep,header=None)
        np_data=np.array(df)
        class_dist=self.get_classname_distribution_from_csv(csv_path=dataset_csv,sep=" ")
        train_count={class_type:0 for (class_type,count) in class_dist}
        val_count = {class_type: 0 for (class_type, count) in class_dist}
        test_count = {class_type: 0 for (class_type, count) in class_dist}
        remain_count = {class_type: 0 for (class_type, count) in class_dist}
        train_list=[]
        val_list=[]
        test_list=[]
        remain_list=[]

        for row in np_data:
            class_id=row[class_type_idx]
            fp=row[0]
            hd,nm=os.path.split(fp)
            nm2=str(nm).split("_")[-1].replace(img_ext,"").replace(".","")
            tc=train_count[nm2]
            vc=val_count[nm2]
            tec=test_count[nm2]
            remc=remain_count[nm2]
            if tec<test_num:
                is_aug_file = False
                for word in test_exclusion_reg:
                    if str(nm).__contains__(word):
                        is_aug_file = True
                        break
                if not is_aug_file:
                    test_list.append(row)
                    tec += 1
                    test_count[nm2] = tec
                else:
                    if tc <train_num:
                        train_list.append(row)
                        tc += 1
                        train_count[nm2] = tc
                    elif vc <val_num:
                        val_list.append(row)
                        vc += 1
                        val_count[nm2] = vc
                    else:
                        continue
            elif vc<val_num:
                # is_aug_file=False
                # for word in val_test_exclusion_reg:
                #     if str(nm).__contains__(word):
                #         is_aug_file=True
                #         break
                # if not is_aug_file:
                val_list.append(row)
                vc+=1
                val_count[nm2]=vc
                # else:
                #     if tc <= train_num:
                #         train_list.append(row)
                #         tc += crossval1.old
                #         train_count[class_id] = tc
            elif  tc<train_num:
                train_list.append(row)
                tc+=1
                train_count[nm2]=tc
            else:#remain data
                remain_list.append(row)
                remc+=1
                remain_count[nm2]=remc

        if len(train_list)>0:
            np.savetxt(train_csv,train_list,fmt="%s",delimiter=sep)
        if len(val_list)>0:
            np.savetxt(val_csv, val_list, fmt="%s", delimiter=sep)
        if len(test_list)>0:
            np.savetxt(test_csv, test_list, fmt="%s", delimiter=sep)
        if len(remain_list)>0:
            if remain_csv is not None:
                np.savetxt(remain_csv, remain_list, fmt="%s", delimiter=sep)
        print "saved %d train %d val, %d test, %d remain" %(len(train_list),len(val_list),len(test_list),len(remain_list))
        return

    # def generate_train_val_by_quantity(self,dataset_csv="/a/b/c.csv",
    #                                             train_csv="/a/b/c.csv",
    #                                             val_csv="/a/b/c.csv",
    #                                             test_csv="/a/b/c.csv",
    #                                             train_num=-crossval1.old,
    #                                             val_num=-crossval1.old,
    #                                             test_num=100,
    #                                             sep=" ",
    #                                             img_ext=".bin.png",
    #                                             class_type_idx=-crossval1.old,
    #                                             fold_num=10,
    #                                             test_exclusion_reg=["_ZOOM","_SHEAR","_TOPLEFTCROP"]):
    #     df=pd.read_csv(dataset_csv,sep=sep,header=None)
    #     np_data=np.array(df)
    #     class_dist=self.get_classname_distribution_from_csv(csv_path=dataset_csv,sep=" ")
    #     train_count={class_type:0 for (class_type,count) in class_dist}
    #     val_count = {class_type: 0 for (class_type, count) in class_dist}
    #     test_count = {class_type: 0 for (class_type, count) in class_dist}
    #     train_list=[]
    #     val_list=[]
    #     test_list=[]
    #
    #     for row in np_data:
    #         class_id=row[class_type_idx]
    #         fp=row[0]
    #         hd,nm=os.path.split(fp)
    #
    #         tc=train_count[class_id]
    #         vc=val_count[class_id]
    #         tec=test_count[class_id]
    #         if tc < train_num:
    #             train_list.append(row)
    #             tc += crossval1.old
    #             train_count[class_id] = tc
    #
    #         elif vc<val_num:
    #             val_list.append(row)
    #             vc+=crossval1.old
    #             val_count[class_id]=vc
    #
    #         else:
    #             test_list.append(row)
    #             tec += crossval1.old
    #             test_count[class_id] = tec
    #
    #     if len(train_list)>0:
    #         np.savetxt(train_csv,train_list,fmt="%s",delimiter=sep)
    #     if len(val_list)>0:
    #         np.savetxt(val_csv, val_list, fmt="%s", delimiter=sep)
    #     if len(test_list)>0:
    #         np.savetxt(test_csv, test_list, fmt="%s", delimiter=sep)
    #     print "saved %d train %d val, %d test" %(len(train_list),len(val_list),len(test_list))
    #     return

    def rename_string_in_dataset_csv(self,dataset_csv="/a/b/c.csv",sep=" ",target_dataset_csv="/a/b/c.csv",string2replace="",replacement_string=""):
        df=pd.read_csv(dataset_csv,sep=sep,header=None)
        np_data=np.array(df.values[:,:])
        modified=[]
        for r in np_data[:]:
            r[0]=str(r[0]).replace(string2replace,replacement_string)
            modified.append(r)
        np.savetxt(target_dataset_csv,modified,fmt="%s",delimiter=" ")
        print "renamed %d rows in dataset"%len(modified)
        return modified

    def check_file_existence_in_dataset_csv(self,dataset_csv="/a/b/c.csv"):
        df = pd.read_csv(dataset_csv,sep=" ",header=None)
        np_data = np.array(df.values[:, :])
        exist=True
        false_file=[]
        for r in np_data:
            if not os.path.exists(r[0]):
                false_file.append(r[0])
                exist=False
        return len(false_file)

    def get_cyclic_lr(self,mode="triangular2",iterations=2000,step_size=2000,base_lr=0.001,max_lr=0.006,gamma=1):
        lr=0.
        if(mode=="triangular"):
            cycle = np.floor(1 + iterations / (2 * step_size))
            x = np.abs(iterations / step_size - 2 * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
        elif (mode=="triangular2"):
            cycle = np.floor(1 + iterations / (2 * step_size))
            x = np.abs(iterations / step_size - 2 * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
        else:
            cycle = np.floor(1 + iterations / (2 * step_size))
            x = np.abs(iterations / step_size - 2 * cycle + 1)
            lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * gamma ** (iterations)
        return lr

    def binarize_img_with_otsu(self,img_path="/a/b/c.png",target_img="/a/b/c.png"):
        img = cv2.imread(img_path, 0)
        ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(target_img, imgf)

    def binarize_dataset_with_otsu(self,src_csv="/a/b/c.csv",sep=" ",filepath_idx=0):
        df=pd.read_csv(filepath_or_buffer=src_csv,sep=sep,header=None)
        np_data=np.array(df)
        i=1
        for row in np_data:
            img_path=row[filepath_idx]
            self.binarize_img_with_otsu(img_path=img_path,target_img=img_path)
            print "%d binarized %s"%(i,img_path)
        print self.get_pixels_binarization_status_from_dataset_csv(csv_path=src_csv,sep=sep,sample_num=100)
        return

    def augment_class_from_csv(self,dataset_csv="/a/b/c.csv",csv_sep=" ", filename_sep="_",img_ext=".bin.png",filepath_idx=0,class_name_idx=-1,class_id=0,dimension_to_crop=[256,256],shear_min=-10,shear_max=10,augment_num=5):
        df=pd.read_csv(filepath_or_buffer=dataset_csv,sep=csv_sep,header=None)
        data_np=np.array(df)
        for row in data_np:
            name=row[class_name_idx]
            if name==class_id:
                filepath = row[filepath_idx]
                img=Image.open(filepath).convert("RGB")
                w, h = img.size
                src_w = w
                src_h = h
                dst_w = dimension_to_crop[0]
                dst_h = dimension_to_crop[1]
                resized_img=img.resize((dst_w,dst_h),Image.ANTIALIAS)
                head,tail = os.path.split(filepath)#
                classn=tail.split(filename_sep)[class_name_idx]
                #classn=classn[0:str(classn).index(img_ext)] #omit extension
                fp_without_class=filepath[0:str(filepath).index(classn)]
                i=0 #begin idx as 2
                for x in range(augment_num):
                    x0=randint(0,dst_w)
                    y0 = randint(0,dst_h)
                    x1 = x0 + (src_w - 1)
                    y1 = y0 + (src_h - 1)
                    if x1 >= dst_w:
                        diff = x1 - dst_w
                        x1 -= diff
                        x0 -= diff
                    if y1 >= dst_h:
                        diff = y1 - dst_h
                        y1 -= diff
                        y0-=diff
                    crop=resized_img.crop((x0,y0,x1,y1))
                    crop_path = fp_without_class + "AUGMENT_"+str(i) + "_" + classn
                    crop.save(crop_path)
                    self.shear_image(img_path=crop_path,target_path=crop_path,shear_min=shear_min,shear_max=shear_max)
                    i += 1
                    print "create crop %s_%d" %(classn,i)
        return

    def augment_class(self,dataset="/a/b/c",csv_sep=" ", filename_sep="_",img_ext=".bin.png",filepath_idx=0,class_name_idx=-1,class_id=0,dimension_to_crop=[256,256],augment_num=5):
        dict=self.get_classname_dictionary(dataset_path=dataset)
        for dp, d, f in os.walk(dataset):
            if len(f) > 0:
                for ff in f:
                    if str(ff).endswith(img_ext):
                        filepath = os.path.join(dp,ff)
                        head, tail = os.path.split(filepath)  #
                        classn = tail.split(filename_sep)[class_name_idx]
                        if classn[:str(classn).index(img_ext)]==dict[class_id]:
                            with Image.open(filepath).convert("RGB") as img:
                                w, h = img.size
                                src_w = w
                                src_h = h
                                dst_w = dimension_to_crop[0]
                                dst_h =dimension_to_crop[1]
                                with img.resize((dst_w, dst_h), Image.ANTIALIAS) as resized_img:
                                        fp_without_class=filepath[0:str(filepath).index(classn)]
                                        i=0 #begin idx as 2
                                        deckw = list(range(1, dst_w))
                                        deckh = list(range(1, dst_h))
                                        random.shuffle(deckw)
                                        random.shuffle(deckh)
                                        for x in range(augment_num):
                                            x0=deckw.pop()
                                            y0 = deckh.pop()
                                            x1 = x0 + (src_w - 1)
                                            y1 = y0 + (src_h - 1)
                                            if x1 >= dst_w:
                                                diff = x1 - dst_w
                                                x1 -= diff
                                                x0 -= diff
                                            if y1 >= dst_h:
                                                diff = y1 - dst_h
                                                y1 -= diff
                                                y0-=diff
                                            with resized_img.crop((x0,y0,x1,y1)) as crop:
                                                crop_path = fp_without_class + "AUGMENT_"+str(i) + "_" + classn
                                                crop.save(crop_path)
                                                crop.close()
                                                self.shear_image(img_path=crop_path,target_path=crop_path)
                                                i += 1
                                                print "create crop %s" %(classn)
        return

    def remove_dataset_images_by_regex(self,dataset="/a/b/c",img_ext=".bin.png",reg="_[0-9]_[A-Za-z\-]+.bin.png"):
        p = re.compile(reg)
        rem = 0
        for dp, d, f in os.walk(dataset):
            if len(f) > 0:
                for ff in f:
                    if str(ff).endswith(img_ext):
                        found = p.findall(ff)
                        if len(found) > 0:
                            fp = os.path.join(dp, ff)
                            os.remove(fp)
                            print "remove %s" % fp
                            rem += 1
        print "total removed:%d" % rem
        return

    def shear_image(self,img_path="/a/b/c.bin.png",target_path="/a/b/c.bin.png",shear_min=-10,shear_max=10):
        # Load the image as a matrix
        #image = Image.open(img_path).convert("RGB")
        image = io.imread(img_path)
        shear = random.uniform(shear_min, shear_max)
        shear=math.radians(shear)
        # Create Afine transform
        afine_tf = tf.AffineTransform(shear=shear)
        # Apply transform to image data
        modified = tf.warp(image, inverse_map=afine_tf)
        io.imsave(target_path, modified)
        with Image.open(target_path).convert("RGBA") as img:
            with Image.new("RGBA",img.size,"white") as  page:
                page.paste(img,(0,0))
                page.save(target_path)
        return

    def generate_crossvalidation_from_csv(self,dataset_csv="/a/b/c.csv",sep=" ",class_type_idx=-1,fold_num=10,target_folder="/a/b/c",test_percentage=0.1):
        #spare the test data
        least_num=self.get_classname_distribution_from_csv(dataset_csv)[-1][1]
        test_csv=os.path.join(target_folder,"crossval_test.csv")
        remain_csv = os.path.join(target_folder, "crossval_remain.csv")
        remain_percentage=1-test_percentage
        self.generate_balanced_train_val_by_percentage(dataset_csv=dataset_csv,
                                                       test_csv=test_csv,
                                                       remain_csv=remain_csv,
                                                       train_percentage=0,
                                                       val_percentage=0,
                                                       test_percentage=test_percentage,
                                                       remain_percentage=remain_percentage)
        df=pd.read_csv(remain_csv,sep=sep,header=None)
        np_xy=np.array(df)
        x=np_xy[:,:class_type_idx]
        y = np_xy[:, class_type_idx]
        head,tail=os.path.split(dataset_csv)
        fn=tail[0:str(tail).index(".csv")]
        kf=KFold(n_splits=fold_num)
        i=1
        for train_idx,val_idx in kf.split(X=np_xy):
            np_train=np_xy[train_idx]
            np_val=np_xy[val_idx]
            train_csv_name=fn+"_train_"+str(i)+".csv"
            val_csv_name = fn + "_val_" + str(i) + ".csv"
            train_fp=os.path.join(target_folder,train_csv_name)
            val_fp = os.path.join(target_folder, val_csv_name)
            np.savetxt(train_fp, np_train, fmt="%s", delimiter=sep)
            np.savetxt(val_fp, np_val, fmt="%s", delimiter=sep)
            print "created crossval %d train:%s val:%s"%(i,train_fp,val_fp)
            i+=1

    def generate_balanced_dataset(self,dataset_csv="/a/b/c.csv",
                                  target_balanced_csv="/a/b/c.csv",
                                  target_remain_csv=None,
                                  sep=" ",
                                  img_ext=".bin.png",
                                  class_type_idx=-1,
                                  num_per_class=None):
        class_dist = self.get_classname_distribution_from_csv(csv_path=dataset_csv, sep=sep)
        if num_per_class is None:
            num_per_class = class_dist[len(class_dist) - 1][1]  # get least number from counter
        df=pd.read_csv(dataset_csv,sep=sep,header=None)
        np_data=np.array(df)
        class_count={class_type:1 for (class_type,count) in class_dist}
        balanced_dataset=[]
        test_dataset=[]
        for row in np_data:
            class_id=row[class_type_idx]
            count=class_count[class_id]
            if  count<=num_per_class:
                balanced_dataset.append(row)
                class_count[class_id]+=1
            else:
                test_dataset.append(row)
        np.savetxt(target_balanced_csv,balanced_dataset,fmt="%s",delimiter=sep)
        if target_remain_csv is not None:
            np.savetxt(target_remain_csv, test_dataset, fmt="%s", delimiter=sep)
        print "saved %d rowed ori_balanced dataset\n%d rowed test dataset" %(len(balanced_dataset),len(test_dataset))
        return balanced_dataset,test_dataset

    def show_cyclical_learning_rate_flow(self,learning_rate = 0.001,max_lr=0.006,num_epochs = 50,train_num = 800,batch_size = 32):
        base_lr = learning_rate
        train_batches_per_epoch = train_num / batch_size
        x = []
        y = []
        for epoch in range(num_epochs):
            for step in range(int(train_batches_per_epoch)):
                iteration = epoch * train_batches_per_epoch + step
                step_size = 5 * train_batches_per_epoch
                learning_rate = self.get_cyclic_lr(mode="triangular2", iterations=iteration, step_size=step_size,
                                                   base_lr=base_lr, max_lr=max_lr)


            x.append(epoch)
            y.append(learning_rate)

        plt.plot(x, y)
        plt.show()

    def remove_dataset_images_by_regex_in_csv(self,
                                              dataset_csv="/a/b/c.csv",
                                              target_csv="/a/b/c.csv",
                                              csv_sep=" ",
                                              img_ext=".bin.png",
                                              reg="_[0-9]_[A-Za-z\-]+.bin.png",
                                              remove_csv_item_only=True):

        #p = re.compile("[A-Za-z\-0-9_]+"+reg+"[A-Za-z\-0-9_]+")
        p = re.compile(reg)
        rem = 0
        noaug=[]
        aug=[]
        df=pd.read_csv(dataset_csv,sep=csv_sep,header=None)
        np_data=np.array(df)
        for row in np_data:
            fp=row[0]
            classid=row[1]
            found = p.findall(fp)
            if len(found) > 0:
                if not remove_csv_item_only:
                    os.remove(fp)
                #aug.append(row)
                rem += 1
            else:
                noaug.append(row)
        if len(noaug)>0:
            np.savetxt(target_csv,noaug,fmt="%s",delimiter=csv_sep)
        print "total filtered out:%d, total saved: %d" % (rem,len(noaug))
        return noaug

    def augment_by_shear_and_zoom(self,
                                    dataset_path="/a/B/c",
                       target_path="/a/b/c",
                       filename_sep="_",
                       img_ext=".bin.png",
                       class_id=0,
                       resize_dim=[256,256],
                       crop_dim=[227,227],
                       shear_min=8,
                       shear_max=8,
                       augment_num=5):
        self.imitate_filetree(src_dataset=dataset_path,dest_dataset=target_path)
        dct=self.get_classname_dictionary(dataset_path)
        temp="/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/temp"
        outdir = os.path.join(temp, "output")
        classname=dct[class_id]
        iteratation=0
        aug=[]
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        fn=str(file).split(filename_sep)[-1]
                        name=fn[:str(fn).index(img_ext)]
                        if name==classname:
                            src= os.path.join(dirpath, file)
                            dst = os.path.join(temp, file)
                            shutil.copyfile(src,dst)

                            #shear
                            pipe = Augmentor.Pipeline(temp)
                            pipe.shear(probability=1,
                                       max_shear_left=shear_min,
                                       max_shear_right=shear_max)
                            #pipe.black_and_white(probability=crossval1.old)
                            for i in range(int(math.ceil(augment_num/2))):
                                pipe.process()
                            i=1
                            os.remove(dst)
                            for dp,ds,fs in os.walk(outdir):
                                if len(fs)>0:
                                    for f in fs:
                                        # fname=str(f).split(filename_sep)[:-2]
                                        # fname="_".join(fname)
                                        f2=f
                                        f2=str(f2).replace(name,"SHEAR"+str(self.get_id())+"_"+name)
                                        f2 = str(f2).replace("temp_original_", "")
                                        addext=str(f2).split("_")[-1]
                                        f2=str(f2).replace("_"+addext,"")
                                        # fname+="_SHEAR"+str(i)+"_"+classname+img_ext
                                        # fname=str(fname).replace("temp_original_","")
                                        src_path=os.path.join(dp,f)
                                        temp_dst=dirpath[len(dataset_path)+1:]
                                        #dst_path=os.path.join(target_path,temp_dst,fname)
                                        dst_path = os.path.join(target_path, temp_dst, f2)
                                        #shutil.copyfile(src=src_path,dst=dst_path)
                                        self.resize_image(src_path,dst_path,crop_dim)
                                        aug.append(dst_path)
                                        os.remove(src_path)
                                        i+=1
                                        iteratation+=1
                                        print "shear %s"%f2
                            #zoom
                            i=1
                            for i in range(int(math.ceil(augment_num/2))):
                                with Image.open(src).convert("RGB") as img:
                                    w,h=img.size
                                    inc=random.randint(5,20)
                                    img=img.resize((crop_dim[0]+inc,crop_dim[1]+inc),Image.ANTIALIAS)
                                    img=img.crop((0,0,crop_dim[0],crop_dim[1]))
                                    f2 = src
                                    f2 = str(f2).replace(name, "ZOOM" + str(self.get_id())+"_"+name)
                                    head,f2=os.path.split(f2)
                                    #f=str(src).replace(classname,"_ZOOM"+str(i)+"_"+classname)
                                    temp_dst = dirpath[len(dataset_path) + 1:]
                                    dst_path = os.path.join(target_path, temp_dst, f2)
                                    img.save(dst_path)
                                    aug.append(dst_path)
                                    #self.binarize_img_with_otsu(dst_path,dst_path)
                                    #self.resize_image(dst_path, dst_path, crop_dim)
                                    iteratation += 1
                                    i+=1
                                    print "zoomed %s" % classname

        print "TOTAL AUGMENTATION class %s : %d"%(classname,iteratation)
        return aug

    def augment_by_orisize_shear_and_zoom(self,
                                    dataset_path="/a/B/c",
                       target_path="/a/b/c",
                       filename_sep="_",
                       img_ext=".bin.png",
                       class_id=0,
                       resize_dim=[256,256],
                       crop_dim=[227,227],
                       shear_min=8,
                       shear_max=8,
                       augment_num=5):
        self.imitate_filetree(src_dataset=dataset_path,dest_dataset=target_path)
        dct=self.get_classname_dictionary(dataset_path)
        temp="/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/temp"
        outdir = os.path.join(temp, "output")
        classname=dct[class_id]
        iteratation=0
        aug=[]
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        fn=str(file).split(filename_sep)[-1]
                        name=fn[:str(fn).index(img_ext)]
                        if name==classname:
                            src= os.path.join(dirpath, file)
                            dst = os.path.join(temp, file)
                            shutil.copyfile(src,dst)

                            #shear
                            pipe = Augmentor.Pipeline(temp)
                            pipe.shear(probability=1,
                                       max_shear_left=shear_min,
                                       max_shear_right=shear_max)
                            #pipe.black_and_white(probability=crossval1.old)
                            for i in range(int(math.ceil(augment_num/2))):
                                pipe.process()
                            i=1
                            os.remove(dst)
                            for dp,ds,fs in os.walk(outdir):
                                if len(fs)>0:
                                    for f in fs:
                                        # fname=str(f).split(filename_sep)[:-2]
                                        # fname="_".join(fname)
                                        f2=f
                                        f2=str(f2).replace(name,"SHEAR"+str(i)+"_"+name)
                                        f2 = str(f2).replace("temp_original_", "")
                                        addext=str(f2).split("_")[-1]
                                        f2=str(f2).replace("_"+addext,"")
                                        # fname+="_SHEAR"+str(i)+"_"+classname+img_ext
                                        # fname=str(fname).replace("temp_original_","")
                                        src_path=os.path.join(dp,f)
                                        temp_dst=dirpath[len(dataset_path)+1:]
                                        #dst_path=os.path.join(target_path,temp_dst,fname)
                                        dst_path = os.path.join(target_path, temp_dst, f2)
                                        shutil.copyfile(src=src_path,dst=dst_path)
                                        #self.resize_image(src_path,dst_path,crop_dim)
                                        aug.append(dst_path)
                                        os.remove(src_path)
                                        i+=1
                                        iteratation+=1
                                        print "shear %s"%f2
                            #zoom
                            i=1
                            for i in range(int(math.ceil(augment_num/2))):
                                with Image.open(src).convert("RGB") as img:
                                    w,h=img.size
                                    inc=random.randint(5,20)
                                    img=img.resize((w+inc,h+inc),Image.ANTIALIAS)
                                    img=img.crop((0,0,w,h))
                                    f2 = src
                                    f2 = str(f2).replace(name, "ZOOM" + str(i)+"_"+name)
                                    head,f2=os.path.split(f2)
                                    #f=str(src).replace(classname,"_ZOOM"+str(i)+"_"+classname)
                                    temp_dst = dirpath[len(dataset_path) + 1:]
                                    dst_path = os.path.join(target_path, temp_dst, f2)
                                    img.save(dst_path)
                                    aug.append(dst_path)
                                    #self.binarize_img_with_otsu(dst_path,dst_path)
                                    #self.resize_image(dst_path, dst_path, crop_dim)
                                    iteratation += 1
                                    i+=1
                                    print "zoomed %s" % classname

        print "TOTAL AUGMENTATION class %s : %d"%(classname,iteratation)
        return aug

    def augment_by_duplicate_images(self,
                                    dataset_path="/a/B/c",
                       target_path="/a/b/c",
                       filename_sep="_",
                       img_ext=".bin.png",
                       class_id=0,
                       augment_num=5):
        self.imitate_filetree(src_dataset=dataset_path,dest_dataset=target_path)
        dct=self.get_classname_dictionary(dataset_path)
        classname=dct[class_id]
        aug=[]
        duplicate_count=1
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        fn=str(file).split(filename_sep)[-1]
                        name=fn[:str(fn).index(img_ext)]
                        if name==classname:
                            for i in range(augment_num):
                                src= os.path.join(dirpath, file)
                                file2=str(file).replace(name,"DUPLICATE"+str(self.get_id())+"_"+name)
                                temp_dst = dirpath[len(dataset_path) + 1:]
                                dst = os.path.join(target_path, temp_dst, file2)
                                shutil.copyfile(src,dst)
                                duplicate_count+=1
                                aug.append(dst)
                                print "COPIED %s"%name
        print "TOTAL AUGMENTATION class %s : %d"%(name,duplicate_count-1)
        return aug



    def get_id(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def resize_image(self,src_img,dst_img,resize_dim=[227,227]):
        with Image.open(src_img).convert("RGB") as img:
            img = img.resize((resize_dim[0], resize_dim[1]), Image.ANTIALIAS)
            img.save(dst_img)

    def binarize_image_with_augmentor(self,src_img,dst_img):
        temp = "/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/temp"
        outdir = os.path.join(temp, "output")
        head,file=os.path.split(src_img)
        dst = os.path.join(temp, file)
        shutil.copyfile(src_img, dst)

        # binarize
        pipe = Augmentor.Pipeline(temp)
        pipe.black_and_white(probability=1)
        pipe.process()
        os.remove(dst)
        head,tail=os.path.split(src_img)
        for dp, ds, fs in os.walk(outdir):
            if len(fs) > 0:
                for f in fs:
                    if str(f).__contains__(tail):
                        src_path = os.path.join(dp, f)
                        shutil.copyfile(src_path, dst_img)
                        os.remove(src_path)
                        print "binarize %s" % dst_img

    def binarize_dataset_from_csv(self,csv_path):
        df=pd.read_csv(csv_path,sep=" ",header=None)
        npd=np.array(df)
        i=0
        for row in npd:
            fp=row[0]
            self.binarize_image_with_augmentor(fp,fp)
            i+=1
        print "binarized %d data"%i

    def binarize_dataset_from_array(self,arr=[]):
        npd=np.array(arr)
        i=0
        for row in npd:
            fp=row
            self.binarize_image_with_augmentor(fp,fp)
            i+=1
        print "binarized %d data"%i

    def prepare_dataset(self,
                        src_dataset_path=["/a/b/c","/a/b/c"],
                        target_augmented_path="/a/b/c",
                        target_augmented_csv="/a/b/c.csv",
                        balanced_csv="/a/b/c.csv",
                        remain_csv="/a/b/c.csv",
                        remain_no_augmentation_csv="/a/b/c.csv",
                        test_plus_remain_no_aug_csv="/a/b/c.csv",
                        train_csv="/a/b/c.csv",
                        val_csv="/a/b/c.csv",
                        test_csv="/a/b/c.csv",
                        train_num=1000,
                        val_num=100,
                        test_num=100,
                        augmented_num_per_class=1500,
                        balanced_class_num=1300,
                        crossval_path="/a/b/c",
                        img_ext=".bin.png",
                        name_splitter="_",
                        resize_dim=[227,227],
                        classnames_to_topleftcrop=["endnote","footnote","footnote-continued"],
                        train_folder="/a/b",
                        val_folder="/a/b",
                        test_folder="/a/b",
                        test_exclude_keyword=["_SHEAR","_ZOOM","_TOPLEFTCROP"],
                        resize_and_copy_source_dataset=True,
                        augmentize=True,
                        binarize=True,
                        use_percentage_for_dataset_generation=False):
        aug_list = []

        if resize_and_copy_source_dataset:
            aug=self.resize_and_copy_to_dataset(dataset_path=src_dataset_path,
                                                       target_path=target_augmented_path,
                                                       resize_dim=resize_dim)
            aug_list += aug
        # # aug = self.copy_to_dataset(dataset_path=src_dataset_path,
        # #                                       target_path=target_augmented_path,
        # #                                       resize_dim=resize_dim)
        # # aug_list+=aug
        # # aug=self.add_topleftcrop_to_dataset(src_dataset_path,
        # #                                 target_path=target_augmented_path,
        # #                                 crop_dim=resize_dim,
        # #                                 classname_to_topleftcrop=classnames_to_topleftcrop)
        # # aug = self.add_orisize_topleftcrop_to_dataset(src_dataset_path,
        # #                                       target_path=target_augmented_path,
        # #                                       crop_dim=resize_dim,
        # #                                       classname_to_topleftcrop=classnames_to_topleftcrop)
        # # aug_list+=aug
        if augmentize:
            dist=self.get_classname_distribution(dataset_path=target_augmented_path)
            dictn=self.get_classname_dictionary(dataset_path=target_augmented_path)
            rdictn={v:k for k,v in dictn.items()}
            classes_to_augment=filter(lambda x:x[1]< augmented_num_per_class,dist) #[x for (x,y) in dist if y<minim_num_per_class]

            for item in classes_to_augment:
                classname=item[0]
                classnum=item[1]
                classid=rdictn[classname]
                augmentnum=int(math.ceil(augmented_num_per_class/classnum))
                aug=self.augment_by_shear_and_zoom(dataset_path=target_augmented_path,
                                    target_path=target_augmented_path,
                                    class_id=classid,augment_num=augmentnum,
                                                   shear_max=2,shear_min=2)
                # aug = self.augment_by_orisize_shear_and_zoom(dataset_path=target_augmented_path,
                #                                      target_path=target_augmented_path,
                #                                      class_id=classid, augment_num=augmentnum,
                #                                      shear_max=2, shear_min=2)

                aug_list+=aug

                print "CLASS %s AUGMENTED to path %s"%(classname,target_augmented_path)
        # aug=self.reshape_relocate_zones_then_merge_to_images(target_augmented_path,target_augmented_path,
        #                                 resize_dim=resize_dim,pad_color="white",relocate=True,
        #                                 average_page_dim=[1776,2546])
        # aug_list+=aug
        #self.invert_database_images(target_augmented_path,target_augmented_path)
        self.convert_dataset_to_csv(dataset_path=target_augmented_path,csv_path=target_augmented_csv)
        #self.convert_dataset_to_csv(dataset_path=src_dataset_path, csv_path=target_augmented_csv)
        # #self.binarize_dataset_from_csv(target_augmented_csv)
        if binarize:
            self.binarize_dataset_from_array(aug_list)
        # # self.generate_balanced_dataset(dataset_csv=target_augmented_csv,
        # #                                target_balanced_csv=balanced_csv,
        # #                                target_remain_csv=remain_csv,
        # #                                num_per_class=balanced_class_num)
        if use_percentage_for_dataset_generation:
            self.generate_balanced_train_val_by_percentage(dataset_csv=target_augmented_csv,
                                                         train_csv=train_csv,
                                                         val_csv=val_csv,
                                                         test_csv=test_csv,
                                                         remain_csv=remain_csv,
                                                         train_percentage=train_num,
                                                           val_percentage=val_num,
                                                           test_percentage=test_num,
                                                         test_exclusion_reg=test_exclude_keyword)
        else:
            self.generate_balanced_train_val_by_quantity(dataset_csv=target_augmented_csv,
                                                           train_csv=train_csv,
                                                           val_csv=val_csv,
                                                           test_csv=test_csv,
                                                           remain_csv=remain_csv,
                                                           train_num=train_num,
                                                           val_num=val_num,
                                                           test_num=test_num,
                                                           test_exclusion_reg=test_exclude_keyword)
        # self.generate_crossvalidation_from_csv(dataset_csv=balanced_csv,
        #                                        fold_num=10,
        #                                        target_folder=crossval_path)
        # self.generate_crossvalidation_from_csv(dataset_csv=target_augmented_csv,
        #                                        fold_num=5,
        #                                        target_folder=crossval_path)

        ss = "("
        for i in range(len(test_exclude_keyword)):
            s = test_exclude_keyword[i]
            if i != len(test_exclude_keyword) - 1:
                ss += s + "|"
            else:
                ss += s
        ss += ")"
        reg = "[A-Za-z\-0-9_]+" + ss + "[A-Za-z\-0-9_]+"
        self.remove_dataset_images_by_regex_in_csv(dataset_csv=remain_csv,
                                                       target_csv=remain_no_augmentation_csv,
                                                       reg=reg)
        self.merge_csvs([test_csv,remain_no_augmentation_csv],test_plus_remain_no_aug_csv)

        return

    def group_images_per_classname(self, dataset_path="/a/b/c", target_path="/a/b/c",img_ext=".bin.png",filename_sep="_",classname_idx=-1,num_per_class=1500,resize_dim=[291,291],is_mini_dataset=False):
        classdict=self.get_classname_dictionary(dataset_path=dataset_path,img_ext=img_ext,filename_sep=filename_sep,classname_idx=classname_idx)
        labels=[v for k,v in classdict.items()]
        counter={k:0 for k in labels}
        for label in labels:
            p=os.path.join(target_path,label)
            if not os.path.isdir(p):
                os.mkdir(p,0777)
        for dp,d,f in os.walk(dataset_path):
            if len(f)>0:
                for file in f:
                    if str(file).endswith(img_ext):
                        classname=str(file).split(filename_sep)[-1]
                        classname=classname[:str(classname).index(img_ext)]
                        if counter[classname]<=num_per_class:
                            src=os.path.join(dp,file)
                            dst=os.path.join(target_path,classname,file)
                            self.resize_binarize_image(src,dst,crop_dim=resize_dim,
                                                       is_mini_dataset=is_mini_dataset)
                            #shutil.copyfile(src,dst)
                            counter[classname]+=1
                            print "copied %s"%classname
                        else:
                            continue

    def rename_to_similarity_group_from_csv(self,src_csv="/a/b.csv",target_csv="/a/b.csv",csv_sep=",",similarity_group=["a_b_c","d_e","f_g"]):
        df=pd.read_csv(src_csv,sep=csv_sep,header=None)
        nparr=np.array(df)
        renamed=0
        temp=[]
        for row in nparr:
            label=row[-1]
            for klas in similarity_group:
                if klas.__contains__(label):
                    row[-1]=klas
                    renamed+=1
                    break
            temp.append(row)
        np.savetxt(target_csv,temp,fmt="%s",delimiter=csv_sep)
        print "renamed %d classes in %s to %s" %(len(temp),src_csv,target_csv)

    def plot_trainval_accloss(self,loss_path="/a/b.csv",acc_path="/a/b.csv",csv_sep=","):
        accdf=pd.read_csv(acc_path,sep=csv_sep,header=None)
        accarr=np.array(accdf)
        acc_epoch=range(1,len(accarr)+1)
        acc_train=accarr[:,1]
        acc_val = accarr[:, -1]

        lossdf = pd.read_csv(loss_path, sep=csv_sep,header=None)
        lossarr = np.array(lossdf)
        loss_epoch = range(1, len(lossarr) + 1)
        loss_train = lossarr[:, 1]
        loss_val = lossarr[:, -1]
        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(loss_epoch,loss_train,"r",loss_epoch,loss_val,"g")
        ax[0].set_title("LOSS")
        #ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].legend(loc="upper right")
        ax[1].plot(acc_epoch, acc_train, "r", acc_epoch, acc_val, "g")
        ax[1].set_title("ACCURACY")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        red_patch = mpatches.Patch(color='red', label='train')
        green_patch = mpatches.Patch(color='green', label='val')
        #blue_patch = mpatches.Patch(color='blue', label='test')
        plt.legend(handles=[red_patch, green_patch])#, blue_patch])
        #ax[1].legend(loc="upper right")
        # for row in ax:
        #     for col in row:
        #         col.plot(x, y)

        plt.show()

    def add_topleftcrop_to_dataset(self,dataset_path="/a/B/c",
                                   target_path="/a/b/c",
                                   filename_sep="_",
                                   img_ext=".bin.png",
                                   crop_dim=[227,227],
                                   classname_to_topleftcrop=["endnote","footnote","footnote-continued"]):
        aug_list=[]
        self.imitate_filetree(src_dataset=dataset_path,dest_dataset=target_path)
        iteratation=0
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        fn=str(file).split(filename_sep)[-1]
                        name=fn[:str(fn).index(img_ext)]
                        if name in classname_to_topleftcrop:
                            src= os.path.join(dirpath, file)
                            with Image.open(src).convert("RGB") as img:
                                w,h=img.size
                                fname=str(file).replace(name,"TOPLEFTCROP_"+name)
                                temp_dst = dirpath[len(dataset_path) + 1:]
                                dst = os.path.join(target_path, temp_dst, fname)
                                #dst = os.path.join(target_path, name, fname)
                                self.crop_topleft_image(src,dst,resize_dim=crop_dim)
                                self.resize_image(dst,dst,crop_dim)
                                aug_list.append(dst)
                                print "crop %s" % fname
                                iteratation+=1
        print "TOTAL TOPLEFTCROPPED class: %d"%(iteratation)
        return aug_list

    def add_orisize_topleftcrop_to_dataset(self,dataset_path="/a/B/c",
                                   target_path="/a/b/c",
                                   filename_sep="_",
                                   img_ext=".bin.png",
                                   crop_dim=[227,227],
                                   classname_to_topleftcrop=["endnote","footnote","footnote-continued"]):
        aug_list=[]
        self.imitate_filetree(src_dataset=dataset_path,dest_dataset=target_path)
        iteratation=0
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        fn=str(file).split(filename_sep)[-1]
                        name=fn[:str(fn).index(img_ext)]
                        if name in classname_to_topleftcrop:
                            src= os.path.join(dirpath, file)
                            with Image.open(src).convert("RGB") as img:
                                w,h=img.size
                                fname=str(file).replace(name,"TOPLEFTCROP_"+name)
                                temp_dst = dirpath[len(dataset_path) + 1:]
                                dst = os.path.join(target_path, temp_dst, fname)
                                #dst = os.path.join(target_path, name, fname)
                                self.crop_topleft_image(src,dst,resize_dim=crop_dim)
                                #self.resize_image(dst,dst,crop_dim)
                                aug_list.append(dst)
                                print "crop %s" % fname
                                iteratation+=1
        print "TOTAL TOPLEFTCROPPED class: %d"%(iteratation)
        return aug_list

    def add_leftcrop_to_dataset(self,dataset_path="/a/B/c",target_path="/a/b/c",filename_sep="_",img_ext=".bin.png",crop_dim=[227,227],classname_to_crop=["endnote","paragraph"]):
        self.imitate_filetree(src_dataset=dataset_path,dest_dataset=target_path)
        iteratation=0
        for dirpath,dirs,files in os.walk(dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        fn=str(file).split(filename_sep)[-1]
                        name=fn[:str(fn).index(img_ext)]
                        #if name in classname_to_crop:
                        src= os.path.join(dirpath, file)
                        with Image.open(src).convert("RGB") as img:
                            w,h=img.size
                            fname = str(file).split(filename_sep)[:-1]
                            fname = "_".join(fname)
                            fname += "_LEFTCROP" + "_" + name + img_ext
                            temp_dst = dirpath[len(dataset_path) + 1:]
                            dst = os.path.join(target_path, temp_dst, fname)
                            #dst = os.path.join(target_path, name, fname)
                            self.crop_topleft_image(src,dst)
                            self.binarize_img_with_otsu(dst,dst)
                            print "crop %s" % fname
                            iteratation+=1
        print "TOTAL TOPLEFTCROPPED class: %d"%(iteratation)
        return

    def prepare_mini_dataset(self,src_dataset_path="/a/b/c",
                        mini_dataset_path="/a/b/c",
                        mini_dataset_csv="/a/b/c.csv",
                        mini_train_csv="/a/b/c.csv",
                        mini_val_csv="/a/b/c.csv",
                        mini_test_csv="/a/b/c.csv",
                        per_class_num=400,
                        mini_train_num=200,
                        mini_val_num=200,
                        img_ext=".bin.png",
                        name_splitter="_",
                             classnames_to_topleftcrop=["endnote"],
                             resize_dim=[227, 227]):
        if os.path.exists(mini_dataset_csv):
            with open(mini_dataset_csv, 'w'): pass
        if os.path.exists(mini_train_csv):
            with open(mini_val_csv, 'w'): pass
        if os.path.exists(mini_val_csv):
            with open(mini_val_csv, 'w'): pass
        if os.path.exists(mini_test_csv):
            with open(mini_train_csv, 'w'): pass

        self.resize_and_add_topleftcrop_to_dataset(dataset_path=src_dataset_path,
                                                   target_path=mini_dataset_path,
                                                   per_class_num=per_class_num,
                                                   classnames_to_topleftcrop=classnames_to_topleftcrop,
                                                   resize_dim=resize_dim)
        dist=self.get_classname_distribution(dataset_path=mini_dataset_path,
                                             img_ext=img_ext,
                                             name_splitter=name_splitter)
        dictn=self.get_classname_dictionary(dataset_path=mini_dataset_path)
        rdictn={v:k for k,v in dictn.items()}
        self.convert_dataset_to_csv(dataset_path=mini_dataset_path,csv_path=mini_dataset_csv)
        self.generate_balanced_train_val_by_quantity(dataset_csv=mini_dataset_csv,
                                                     train_csv=mini_train_csv,
                                                     val_csv=mini_val_csv,
                                                     train_num=mini_train_num,
                                                     val_num=mini_val_num,
                                                     test_csv=mini_test_csv,
                                                     classnames_to_topleftcrop=classnames_to_topleftcrop)


    def resize_and_copy_to_dataset(self,
                                                       dataset_path="/a/B/c",
                                                       target_path="/a/b/c",
                                                       filename_sep="_",
                                                       img_ext=".bin.png",
                                                       resize_dim=[227, 227]):
        aug_list=[]
        self.imitate_filetree(src_dataset=dataset_path, dest_dataset=target_path)
        dictn=self.get_classname_dictionary(dataset_path)#{0: 'paragraph', crossval1.old: 'page-number'
        iteratation = 0
        for dirpath, dirs, files in os.walk(dataset_path):
            if len(files) > 0:
                for file in files:
                    if str(file).endswith(img_ext):
                        fn = str(file).split(filename_sep)[-1]
                        name = fn[:str(fn).index(img_ext)]
                        #if counter[name]<=per_class_num:
                        src = os.path.join(dirpath, file)
                        imgg = Image.open(src)
                        w,h=imgg.size
                        tmp=dirpath[len(dataset_path)+1:]
                        dst=os.path.join(target_path,tmp,file.replace(name,"RESIZED_"+name))
                        with Image.open(src).convert("RGB") as img:
                            resized_img = img.resize((resize_dim[0], resize_dim[1]), Image.ANTIALIAS)
                            #resized_img = resized_img.crop((0, 0, resize_dim[0], resize_dim[crossval1.old]))
                            resized_img.save(dst)
                            aug_list.append(dst)
                            print "resize %s"%name
                            iteratation+=1
        print "TOTAL Processed class: %d" % (iteratation)
        return aug_list

    def copy_to_dataset(self,
                                                       dataset_path="/a/B/c",
                                                       target_path="/a/b/c",
                                                       filename_sep="_",
                                                       img_ext=".bin.png",
                                                       resize_dim=[227, 227]):
        aug_list=[]
        self.imitate_filetree(src_dataset=dataset_path, dest_dataset=target_path)
        dictn=self.get_classname_dictionary(dataset_path)#{0: 'paragraph', crossval1.old: 'page-number'
        counter={v:0 for k,v in dictn.items()}
        iteratation = 0
        for dirpath, dirs, files in os.walk(dataset_path):
            if len(files) > 0:
                for file in files:
                    if str(file).endswith(img_ext):
                        fn = str(file).split(filename_sep)[-1]
                        name = fn[:str(fn).index(img_ext)]
                        #if counter[name]<=per_class_num:
                        src = os.path.join(dirpath, file)
                        imgg = Image.open(src)
                        w,h=imgg.size
                        tmp=dirpath[len(dataset_path)+1:]
                        dst=os.path.join(target_path,tmp,file.replace(name,"RESIZED_"+name))
                        with Image.open(src).convert("RGB") as img:
                            img.save(dst)
                            #self.binarize_img_with_otsu(dst,dst)
                            aug_list.append(dst)
                            print "resize %s"%name
                            iteratation+=1
        print "TOTAL Processed class: %d" % (iteratation)
        return aug_list

    def get_feature_map_dim_before_global_avg(self,height=227,width=227,kernels=[11,3,3],strides=[4,2,2]):
        h=height
        w=width
        hs=[]
        ws=[]
        for i in range(len(kernels)):
            h=((h-kernels[i])/strides[i])+1
            w = ((w - kernels[i]) / strides[i]) + 1
            hs.append(h)
            ws.append(w)
        return hs,ws

    def get_convolvable_dim(self,h,w,kernels=[11,3,3],strides=[4,2,2]):
        reth=h
        retw=w
        lastw=0
        lasth=0
        h_found=False
        w_found=False
        while not h_found and not w_found:
            hh, ww = self.get_feature_map_dim_before_global_avg(h, w,kernels,strides)
            if hh[-1].is_integer():
                reth=h
                lasth=hh
                h_found=True
            else:
                h -= 1
            if ww[-1].is_integer():
                retw=w
                lastw=ww
                w_found=True
            else:
                w -= 1

        return reth,retw,lasth,lastw



    # def generate_balanced_train_val_by_quantity2(self,dataset_path="/a/b/c",train_path="/a/b/c",val_path="/a/b/c",test_path="/a/b/c",train_num=200,val_num=100,test_num=100,sep=" ", img_ext=".bin.png",class_type_idx=-crossval1.old,resize_dim=[291,291],val_and_test_exclude_keword="CROP"):
    #     #df=pd.read_csv(dataset_csv,sep=sep)
    #     #np_data=np.array(df)
    #     self.imitate_filetree(dataset_path,train_path)
    #     self.imitate_filetree(dataset_path, val_path)
    #     self.imitate_filetree(dataset_path, test_path)
    #     dct=self.get_classname_dictionary(dataset_path)
    #     dct={v:k for k,v in dct.items()}
    #     class_dist=self.get_classname_distribution(dataset_path)
    #     train_count={class_type:crossval1.old for (class_type,count) in class_dist}
    #     val_count = {class_type: crossval1.old for (class_type, count) in class_dist}
    #     test_count = {class_type: crossval1.old for (class_type, count) in class_dist}
    #     train_list=[]
    #     val_list=[]
    #     test_list=[]
    #
    #     for dirpath,dirs,files in os.walk(dataset_path):
    #         if len(files)>0:
    #             for file in files:
    #                 if str(file).endswith(img_ext):
    #                     class_id=str(file).split("_")[-crossval1.old]
    #                     class_id=class_id[:str(class_id).index(img_ext)]
    #                     #class_id=dct[class_id]
    #                     tc=train_count[class_id]
    #                     vc=val_count[class_id]
    #                     tec=test_count[class_id]
    #                     srcp=os.path.join(dirpath,file)
    #                     tp=dirpath[len(dataset_path)+crossval1.old:]
    #                     if  tc<=train_num:
    #
    #                         trp = os.path.join(train_path,tp,file)
    #                         shutil.copyfile(srcp,trp)
    #                         tc+=crossval1.old
    #                         train_count[class_id]=tc
    #
    #                     elif vc<=val_num:
    #                         if str(file).__contains__(val_and_test_exclude_keword):
    #                             continue
    #                         valp = os.path.join(val_path,tp, file)
    #                         shutil.copyfile(srcp, valp)
    #                         vc+=crossval1.old
    #                         val_count[class_id]=vc
    #
    #                     elif tec<test_num:
    #                         if str(file).__contains__(val_and_test_exclude_keword):
    #                             continue
    #                         tsp = os.path.join(test_path, tp,file)
    #                         shutil.copyfile(srcp,tsp)
    #                         tec+=crossval1.old
    #                         test_count[class_id]=tec
    #                     else:
    #                         break
    #
    #
    #     print "saved %d train %d val, %d test" %(tc,vc,tec)
    #     return

    def resize_binarize_image(self,
                       src_path="/a/B/c.png",
                       target_path="/a/b/c.png",
                       filename_sep="_",
                       img_ext=".bin.png",
                       crop_dim=[224, 224],
                       shear_min=5,
                       shear_max=5,
                              is_mini_dataset=False):
        temp = "/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/temp"
        outdir = os.path.join(temp, "output")
        src = src_path
        head,file=os.path.split(src)
        name=str(file).split("_")[-1]
        name=name[:str(name).index(img_ext)]
        imgg = Image.open(src)
        w,h=imgg.size
        dst = os.path.join(temp, file)
        # if is_mini_dataset:
        #     with Image.open(src).convert("RGB") as imm:
        #         imm=imm.resize(crop_dim[0]+10,crop_dim[crossval1.old]+10)
        #         imm=imm.crop((0,0,w,h))
        #         imm.save(src)
        shutil.copyfile(src, dst)
        pipe = Augmentor.Pipeline(temp)
        #wh=crop_dim[0]+int(math.ceil(crop_dim[0]*crossval1.old.3))
        wh=crop_dim[0]
        pipe.resize(probability=1, width=wh, height=wh)
        #pipe.crop_by_size(probability=crossval1.old, width=crop_dim[0], height=crop_dim[crossval1.old], centre=False)
        #pipe.shear(probability=crossval1.old, max_shear_left=shear_min, max_shear_right=shear_max)
        pipe.black_and_white(probability=1)
        pipe.process()
        os.remove(dst)
        i=1
        for dp,ds,fs in os.walk(outdir):
            if len(fs)>0:
                for f in fs:
                    fname=str(f).split(filename_sep)[:-2]
                    fname="_".join(fname)
                    fname+="_RESIZED"+str(i)+"_"+name+img_ext
                    fname=str(fname).replace("temp_original_","")
                    src_path=os.path.join(dp,f)
                    dst_path=target_path#os.path.join(target_path,temp_dst,fname)
                    shutil.copyfile(src=src_path,dst=dst_path)
                    os.remove(src_path)
                    i+=1
                    print "created %s"%fname
        return

    def get_branch_channel(self,fold=6, out_channel=2048):
        per_branch_ch = int(out_channel / fold)
        remain = out_channel % per_branch_ch
        ret = []
        for i in range(fold):
            if i == fold - 1:
                ret.append(per_branch_ch + remain)
            else:
                ret.append(per_branch_ch)
        return ret

    def crop_topleft_image(self,
                       src_path="/a/B/c.png",
                       target_path="/a/b/c.png",
                       filename_sep="_",
                       img_ext=".bin.png",
                       resize_dim=[227, 227],
                       classnames_to_topleftcrop=["endnote","footnote"]):
        src = src_path
        head,file=os.path.split(src)
        name=str(file).split("_")[-1]
        name=name[:str(name).index(img_ext)]
        with Image.open(src).convert("RGB") as img:
            w,h=img.size
            img=img.crop((0,0,int(w/2),int(h/2)))
            img.save(target_path)
            #self.binarize_img_with_otsu(target_path,target_path)
            #print "TOPLEFTCROP to: %s"%target_path
        return




    def prepare_data_by_grouping(self,
                                 dataset_path,
                                 target_path,
                                 num_per_class,
                                 resize_dim,
                                 classnames_to_topleftcrop,
                                 train_folder,
                                 val_folder,
                                 test_folder,
                                 train_num,
                                 val_num,
                                 test_num,
                                 val_and_test_exclude_keyword,
                                 train_csv,
                                 val_csv,
                                 test_csv,
                                 dataset_csv):
        self.group_images_per_classname(dataset_path=dataset_path,
                                        target_path=target_path,
                                        num_per_class=num_per_class,
                                        resize_dim=resize_dim
                                        )
        self.add_topleftcrop_to_dataset(dataset_path,
                                        target_path=target_path,
                                        crop_dim=resize_dim,
                                        classname_to_crop=classnames_to_topleftcrop)
        self.generate_balanced_train_val_by_quantity2(dataset_path=target_path,
                                                      train_path=train_folder,
                                                      val_path=val_folder,
                                                      test_path=test_folder,
                                                      train_num=train_num,
                                                      val_num=val_num,
                                                      test_num=test_num,
                                                      resize_dim=resize_dim,
                                                      val_and_test_exclude_keword=val_and_test_exclude_keyword)
        self.convert_dataset_to_csv(dataset_path=target_path, csv_path=dataset_csv)
        self.convert_dataset_to_csv(dataset_path=train_folder,csv_path=train_csv)
        self.convert_dataset_to_csv(dataset_path=val_folder,csv_path=val_csv)
        self.convert_dataset_to_csv(dataset_path=test_folder,csv_path=test_csv)


    def load_tf_checkpoint(self,ckpt_path="/a/b.ckpt"):
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        dicti= {k:reader.get_tensor(k) for k in sorted(var_to_shape_map)}
        return dicti

    def get_tf_checkpoint_by_tensorname(self,ckpt_path="/a/b.ckpt",tname="conv4"):
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        if tname in var_to_shape_map:
            tensor=reader.get_tensor(tname) #h,w,prev channel,next channel
            return tensor
        else:
            return ""

    def select_unaugmented_data_to_test_data(self,val_csv,test_csv,augmentation_keyword=["_SHEAR","_ZOOM","_TOPLEFTCROP"]):
        return

    def invert_database_images(self,src_database_path,dst_database_path,img_ext=".bin.png"):
        self.imitate_filetree(src_database_path,dst_database_path)
        count=0
        for dirpath,dirs,files in os.walk(src_database_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        impth=os.path.join(dirpath,file)
                        tmpdir=dirpath[len(src_database_path)+1:]
                        with Image.open(impth) as image:
                            try:
                                inverted_image = PIL.ImageOps.invert(image)
                                head, tail = os.path.split(impth)
                                fn = tail.split("_")[-1]
                                #tail=tail.replace(fn,"INV_"+fn)
                                impth2 = os.path.join(dst_database_path, tmpdir,tail)
                                inverted_image.save(impth2)
                                count+=1
                                print "CONVERTED:\r%s"%fn
                            except Exception,e:
                                print "ERROR:%s"%str(e)
                                print "IMAGE ERROR:%s"%impth
        print "INVERTED %d images"%count


    def prepare_dataset_by_reloc_and_unreloc(self,
                        src_dataset_path=["/a/b/c","/a/b/c"],
                        target_augmented_path="/a/b/c",
                        target_augmented_csv="/a/b/c.csv",
                        balanced_csv="/a/b/c.csv",
                        remain_csv="/a/b/c.csv",
                        remain_no_augmentation_csv="/a/b/c.csv",
                        balanced_train_csv="/a/b/c.csv",
                        balanced_val_csv="/a/b/c.csv",
                        balanced_test_csv="/a/b/c.csv",
                        balanced_train_num=1000,
                        balanced_val_num=100,
                        balanced_test_num=100,
                        augmented_num_per_class=1500,
                        balanced_class_num=1300,
                        crossval_path="/a/b/c",
                        img_ext=".bin.png",
                        name_splitter="_",
                        resize_dim=[227,227],
                        classnames_to_topleftcrop=["endnote","footnote","footnote-continued"],
                        train_folder="/a/b",
                        val_folder="/a/b",
                        test_folder="/a/b",
                        test_exclude_keyword=["_SHEAR","_ZOOM","_TOPLEFTCROP"]):
        aug_list = []
        aug=self.resize_and_copy_to_dataset(dataset_path=src_dataset_path,
                                                   target_path=target_augmented_path,
                                                   resize_dim=resize_dim)
        aug_list += aug
        # aug = self.copy_to_dataset(dataset_path=src_dataset_path,
        #                                       target_path=target_augmented_path,
        #                                       resize_dim=resize_dim)
        #aug_list+=aug
        aug=self.add_topleftcrop_to_dataset(src_dataset_path,
                                        target_path=target_augmented_path,
                                        crop_dim=resize_dim,
                                        classname_to_topleftcrop=classnames_to_topleftcrop)
        aug_list += aug
        # aug = self.add_orisize_topleftcrop_to_dataset(src_dataset_path,
        #                                       target_path=target_augmented_path,
        #                                       crop_dim=resize_dim,
        #                                       classname_to_topleftcrop=classnames_to_topleftcrop)
        # aug_list+=aug
        dist=self.get_classname_distribution(dataset_path=target_augmented_path)
        dictn=self.get_classname_dictionary(dataset_path=target_augmented_path)
        rdictn={v:k for k,v in dictn.items()}
        classes_to_augment=filter(lambda x:x[1]< augmented_num_per_class,dist) #[x for (x,y) in dist if y<minim_num_per_class]
        for item in classes_to_augment:
            classname=item[0]
            classnum=item[1]
            classid=rdictn[classname]
            augmentnum=int(math.ceil(augmented_num_per_class/classnum))
            aug=self.augment_by_shear_and_zoom(dataset_path=target_augmented_path,
                                target_path=target_augmented_path,
                                class_id=classid,augment_num=augmentnum,
                                               shear_max=5,shear_min=5)
            # aug = self.augment_by_orisize_shear_and_zoom(dataset_path=target_augmented_path,
            #                                      target_path=target_augmented_path,
            #                                      class_id=classid, augment_num=augmentnum,
            #                                      shear_max=5, shear_min=5)
            aug_list+=aug
            print "CLASS %s AUGMENTED to path %s"%(classname,target_augmented_path)

        aug = self.reshape_and_relocate_zones(target_augmented_path, target_augmented_path,
                                              resize_dim=resize_dim, pad_color="white", relocate=True,
                                              average_page_dim=[1776, 2546])

        #self.invert_database_images(target_augmented_path,target_augmented_path)
        self.convert_dataset_to_csv(dataset_path=target_augmented_path,csv_path=target_augmented_csv)
        #self.binarize_dataset_from_csv(target_augmented_csv)
        self.binarize_dataset_from_array(aug_list)
        self.generate_balanced_dataset(dataset_csv=target_augmented_csv,
                                       target_balanced_csv=balanced_csv,
                                       target_remain_csv=remain_csv,
                                       num_per_class=balanced_class_num)

        self.generate_balanced_train_val_by_quantity(dataset_csv=target_augmented_csv,
                                                     train_csv=balanced_train_csv,
                                                     val_csv=balanced_val_csv,
                                                     test_csv=balanced_test_csv,
                                                     train_num=balanced_train_num,
                                                     val_num=balanced_val_num,
                                                     test_num=balanced_test_num,
                                                     test_exclusion_reg=test_exclude_keyword)
        # self.generate_crossvalidation_from_csv(dataset_csv=balanced_csv,
        #                                        fold_num=10,
        #                                        target_folder=crossval_path)
        for s in test_exclude_keyword:
            self.remove_dataset_images_by_regex_in_csv(dataset_csv=remain_csv,
                                                       target_csv=remain_no_augmentation_csv,
                                                       reg=s)
        return


    def merge_reloc_and_unreloc_image(self,img1,img2,img3,dst_img):
        np_im1 = cv2.imread(img1)
        np_im1 = np_im1[:, :, :1]

        np_im2 = cv2.imread(img2)
        np_im2 = np_im2[:, :, :1]

        np_im3 = cv2.imread(img3)
        np_im3 = np_im3[:, :, :1]

        np_im4 = np.concatenate((np_im1, np_im2,np_im3), axis=2)
        cv2.imwrite(dst_img, np_im4)
        return

    def check_image_channel_homogenity(self, np_arr):
        homogen=True
        for y in range(len(np_arr[:,:,:])):
            for x in range(len(np_arr[y,:,:])):
                b=np_arr[y,x,0]
                g = np_arr[y, x, 1]
                r = np_arr[y, x, 2]
                if not (b==g==r):
                    homogen=False
                    break
        return homogen


    def copy_similar_class_between_dataset(self,src_dataset="/a",dst_dataset="/a",img_ext=".bin.png",class_to_copy=["a"]):
        copied=[]
        self.imitate_filetree(src_dataset,dst_dataset)
        for dp,d,f in os.walk(src_dataset):
            if len(f)>0:
                for file in f:
                    if str(file).endswith(img_ext):
                        fn=str(file).split("_")[-1].replace(img_ext,"")
                        if fn in class_to_copy:
                            srcp=os.path.join(dp,file)
                            tmpd=dp[len(src_dataset)+1:]
                            dstp=os.path.join(dst_dataset,tmpd,file)
                            shutil.copyfile(srcp,dstp)
                            print "COPIED:%s"%fn
                            copied.append(dstp)
        return copied

    def augment_dataset_by_filepaths(self,
                        files_to_augment=["/a.bin.png","/b.bin.png"],
                        src_dataset_path=["/a/b/c","/a/b/c"],
                        target_augmented_path="/a/b/c",
                        target_augmented_csv="/a/b/c.csv",
                        balanced_csv="/a/b/c.csv",
                        remain_csv="/a/b/c.csv",
                        remain_no_augmentation_csv="/a/b/c.csv",
                        balanced_train_csv="/a/b/c.csv",
                        balanced_val_csv="/a/b/c.csv",
                        balanced_test_csv="/a/b/c.csv",
                        balanced_train_num=1000,
                        balanced_val_num=100,
                        balanced_test_num=100,
                        augmented_num_per_class=1500,
                        balanced_class_num=1300,
                        crossval_path="/a/b/c",
                        img_ext=".bin.png",
                        name_splitter="_",
                        resize_dim=[227,227],
                        classnames_to_topleftcrop=["endnote","footnote","footnote-continued"],
                        train_folder="/a/b",
                        val_folder="/a/b",
                        test_folder="/a/b",
                        test_exclude_keyword=["_SHEAR","_ZOOM","_TOPLEFTCROP"]):
        aug_list = []

        aug = self.add_orisize_topleftcrop_by_filepaths(filepaths=files_to_augment,
                                              crop_dim=resize_dim,
                                              classname_to_topleftcrop=classnames_to_topleftcrop)
        aug_list+=aug
        dist=self.get_classname_distribution(dataset_path=target_augmented_path)
        dictn=self.get_classname_dictionary(dataset_path=target_augmented_path)
        rdictn={v:k for k,v in dictn.items()}
        classes_to_augment=filter(lambda x:x[1]< augmented_num_per_class,dist) #[x for (x,y) in dist if y<minim_num_per_class]
        for item in classes_to_augment:
            classname=item[0]
            classnum=item[1]
            classid=rdictn[classname]
            augmentnum=int(math.ceil(augmented_num_per_class/classnum))
            # aug=self.augment_by_shear_and_zoom(dataset_path=target_augmented_path,
            #                     target_path=target_augmented_path,
            #                     class_id=classid,augment_num=augmentnum,
            #                                    shear_max=5,shear_min=5)
            aug = self.augment_by_orisize_shear_and_zoom(dataset_path=target_augmented_path,
                                                 target_path=target_augmented_path,
                                                 class_id=classid, augment_num=augmentnum,
                                                 shear_max=5, shear_min=5)
            aug_list+=aug
            print "CLASS %s AUGMENTED to path %s"%(classname,target_augmented_path)
        aug=self.reshape_relocate_zones_then_merge_to_images(target_augmented_path,target_augmented_path,
                                        resize_dim=resize_dim,pad_color="white",relocate=True,
                                        average_page_dim=[1776,2546])
        aug_list+=aug
        #self.invert_database_images(target_augmented_path,target_augmented_path)
        self.convert_dataset_to_csv(dataset_path=target_augmented_path,csv_path=target_augmented_csv)
        #self.binarize_dataset_from_csv(target_augmented_csv)
        #self.binarize_dataset_from_array(aug_list)
        self.generate_balanced_dataset(dataset_csv=target_augmented_csv,
                                       target_balanced_csv=balanced_csv,
                                       target_remain_csv=remain_csv,
                                       num_per_class=balanced_class_num)

        self.generate_balanced_train_val_by_quantity(dataset_csv=target_augmented_csv,
                                                     train_csv=balanced_train_csv,
                                                     val_csv=balanced_val_csv,
                                                     test_csv=balanced_test_csv,
                                                     train_num=balanced_train_num,
                                                     val_num=balanced_val_num,
                                                     test_num=balanced_test_num,
                                                     test_exclusion_reg=test_exclude_keyword)
        # self.generate_crossvalidation_from_csv(dataset_csv=balanced_csv,
        #                                        fold_num=10,
        #                                        target_folder=crossval_path)
        for s in test_exclude_keyword:
            self.remove_dataset_images_by_regex_in_csv(dataset_csv=remain_csv,
                                                       target_csv=remain_no_augmentation_csv,
                                                       reg=s)
        return


    def add_orisize_topleftcrop_by_filepaths(self,
                                             filepaths=["/a/B/c.bin.png"],
                                   filename_splitter="_",
                                   filename_sep="_",
                                   img_ext=".bin.png",
                                   crop_dim=[227,227],
                                   classname_to_topleftcrop=["endnote","footnote","footnote-continued"]):
        aug_list=[]
        iteration=0
        for file in filepaths:
            name = str(file).split(filename_splitter)[-1].replace(img_ext, "")
            if name in classname_to_topleftcrop:
                with Image.open(file).convert("RGB") as img:
                    w,h=img.size
                    dstfp=str(file).replace(name,"TOPLEFTCROP_"+name)
                    self.crop_topleft_image(file,dstfp,resize_dim=crop_dim)
                    aug_list.append(dstfp)
                    print "crop %s" % name
                    iteration+=1
        print "TOTAL TOPLEFTCROPPED class: %d"%(iteration)
        return aug_list

    def prepend_text_to_dataset_csv(self,csv_file="/a.csv",prependstr="/home/",csv_sep=" "):
        train = csv_file
        pre = prependstr
        df = pd.read_csv(train, sep=csv_file,engine='python',header=None)
        npdf = np.array(df)
        for row in npdf:
            row[0] = pre + row[0]
        np.savetxt(train, npdf, delimiter=csv_sep, fmt="%s")
        print "Prepended %d to %s"%(len(npdf),train)
        return

    def convert_csv_to_dictionary(self,csv_file,csv_sep=" ",name_label=-1,name_idx=0):
        df=pd.read_csv(csv_file,sep=csv_sep,header=None)
        npdf=np.array(df)
        dct={}
        for row in npdf:
            dct[row[name_idx]]=row[name_label]
        return dct

    def convert_tif_to_jpg(self,src_dataset="/a",dst_dataset="/b",resize_dim=[227,227]):
        #yourpath = os.getcwd()
        self.imitate_filetree(src_dataset,dst_dataset)
        for root, dirs, files in os.walk(src_dataset, topdown=False):
            for name in files:
                print(os.path.join(root, name))
                if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
                    if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                        print "A jpeg file already exists for %s" % name
                    # If a jpeg is *NOT* present, create one from the tiff.
                    else:
                        tmpdir=root[len(src_dataset)+1:]
                        tmpdir=os.path.join(dst_dataset,tmpdir)
                        outfile = os.path.splitext(os.path.join(tmpdir, name))[0] + ".jpg"
                        try:
                            with Image.open(os.path.join(root, name)) as im:
                                print "Generating jpeg for %s" % name
                                im=im.resize(resize_dim, Image.ANTIALIAS)
                                im.save(outfile, "JPEG", quality=100)
                        except Exception, e:
                            print "ERROR:%s"%str(e)
        return

    def copy_images_between_datasets(self,src_dataset,dst_dataset,img_ext=".bin.png",
                                     filename_sep="_",
                                     classname_to_copy=["a","b"],
                                     classname_after_renamed=["aa","bb"]):
        iterate=0
        self.imitate_filetree(src_dataset,dst_dataset)
        classname_to_copy=[c.lower() for c in classname_to_copy]
        for dirpath,dirs,files in os.walk(src_dataset):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        name=str(file).split(filename_sep)[-1].replace(img_ext,"")
                        if name.lower() in classname_to_copy:
                            idx=classname_to_copy.index(name.lower())
                            replacement_name=classname_after_renamed[idx]
                            replacement_file=str(file).replace(name,replacement_name)
                            src=os.path.join(dirpath,file)
                            tmpdir=dirpath[len(src_dataset)+1:]
                            dst=os.path.join(dst_dataset,tmpdir,replacement_file)
                            shutil.copyfile(src,dst)
                            print("COPIED:\r%s"%name)
        return

    def convert_array_to_csv(self,arr=[],
                               csv_path="/a/b/c.csv",
                               csv_sep=" ",
                               filename_sep="_",
                               filepath_idx=0,
                               filename_class_idx=-1):
        np.savetxt(csv_path,arr,fmt="%s",delimiter=csv_sep)
        print "CONVERTED %d to %s"%(len(arr),csv_path)
        return

    def merge_csvs(self,csv_paths=[],csvmerge_path="/a.csv",sep=" ",img_ext=".bin.png"):
        l=0
        with open(csvmerge_path,"w+") as file:
            writer = csv.writer(file,delimiter =" ")
            for i in range(len(csv_paths)):
                df=pd.read_csv(csv_paths[i],sep=sep)
                npd=np.array(df)
                for row in npd:
                    row = row.tolist()
                    writer.writerow(row)
                l+=len(npd)
        print "SAVED %d rows to %s"%(l,csvmerge_path)
        return

    def get_classname_set_from_csv(self,dataset_csv,img_ext=".bin.png",sep=" ",classname_idx=-1):
        df=pd.read_csv(dataset_csv,sep=sep)
        npd=np.array(df)
        pairs=[]
        for row in npd:
            x=row[0]
            y=row[classname_idx]
            clasname=str(x).split("_")[-1].replace(img_ext,"")
            pairs.append([y,  clasname])
        pairs=[list(x) for x in set(tuple(x) for x in pairs)]
        class_dictionary = pairs

        return class_dictionary


    def get_database_average_pixel_from_csv(self,dataset_csv="/a/b/c.csv",csv_sep=" ",img_ext=".bin.png"):
        pix_val=[0,0,0]
        pix_count=0
        i=0
        df=pd.read_csv(dataset_csv,sep=csv_sep)
        npd=np.array(df)
        for row in npd:
             imgpath=row[0]
             if str(imgpath).endswith(img_ext):
                try:
                    image = cv2.imread(imgpath)
                    rows, cols, channels = image.shape
                    pix_count += (rows * cols)
                    for x in range(rows):
                        for y in range(cols):
                            pix_val[0] += image[x, y, 0]
                            # pix_val[crossval1.old] += image[x, y, crossval1.old]
                            # pix_val[2] += image[x, y, 2]

                    #print "%d/400000"%i
                    i+=1
                except Exception,e:
                    print "ERROR:%s"%file

        avg_pix=[p/pix_count for p in pix_val]#pix_val/pix_count
        return avg_pix


    def get_database_average_pixel_by_tensorflow(self,dataset_path="/a/b/c",img_ext=".jpg"):
        pix_val = [0, 0, 0]
        pix_count = 0
        i = 0
        avg=0.0
        img_concat=np.zeros([227,227,3],float)
        with tflow.Session() as sess:
                for dirpath,dirs,files in os.walk(dataset_path):
                    if len(files)>0:
                        for file in files:
                            if str(file).endswith(img_ext):
                                try:
                                    filepath = os.path.join(dirpath, file)
                                    image = cv2.imread(filepath)
                                    img_concat=np.concatenate((img_concat,image),2)

                                    # rows, cols, channels = image.shape
                                    # pix_count += (rows * cols)
                                    # for x in range(rows):
                                    #     for y in range(cols):
                                    #         pix_val[0] += image[x, y, 0]
                                    #         # pix_val[crossval1.old] += image[x, y, crossval1.old]
                                    #         # pix_val[2] += image[x, y, 2]
                                    #         print pix_val[0]/pix_count
                                    # #print "%d/400000"%i
                                    print "\r%d of 519833, %f"%(i,i/519833)
                                    i+=1
                                except Exception,e:
                                    print "ERROR:%s"%file
                avg_op = tflow.reduce_mean(img_concat)
                avg = sess.run(avg_op)
                print "AVG CONCAT:%.5f"%avg
                #avg_pix=[p/pix_count for p in pix_val]#pix_val/pix_count
        return avg

    def generate_balanced_train_val_by_percentage(self,dataset_csv="/a/b/c.csv",
                                                train_csv="/a/b/c.csv",
                                                val_csv="/a/b/c.csv",
                                                test_csv="/a/b/c.csv",
                                                remain_csv=None,
                                                train_percentage=0.7,
                                                val_percentage=0.15,
                                                test_percentage=0.15,
                                                remain_percentage=0.0,
                                                sep=" ",
                                                img_ext=".bin.png",
                                                class_type_idx=-1,
                                                #fold_num=10,
                                                test_exclusion_reg=["_ZOOM","_SHEAR","_TOPLEFTCROP"],
                                                class_id_is_label=False):
        with open(train_csv,"a+") as f:
            pass
        with open(test_csv,"a+") as f:
            pass
        df=pd.read_csv(dataset_csv,sep=sep,header=None)
        np_data=np.array(df)
        class_dist=self.get_classname_distribution_from_csv(csv_path=dataset_csv,sep=sep,
                                                            class_id_is_label=class_id_is_label)
        train_count={class_type:0 for (class_type,count) in class_dist}
        val_count = {class_type: 0 for (class_type, count) in class_dist}
        test_count = {class_type: 0 for (class_type, count) in class_dist}
        remain_count = {class_type: 0 for (class_type, count) in class_dist}

        train_nums = {class_type: int(math.ceil(train_percentage*count)) for (class_type, count) in class_dist}
        val_nums = {class_type: int(math.ceil(val_percentage*count)) for (class_type, count) in class_dist}
        test_nums = {class_type: int(math.ceil(test_percentage*count)) for (class_type, count) in class_dist}
        remain_nums = {class_type: int(math.ceil(remain_percentage*count)) for (class_type, count) in class_dist}
        train_list=[]
        val_list=[]
        test_list=[]
        remain_list=[]

        for row in np_data:
            class_id=row[class_type_idx]
            nm=""
            try:
                fp=row[0]
                hd,nm=os.path.split(fp)
                nm=str(nm).split("_")[-1].replace(img_ext,"")
            except Exception,e:
                pass
            if class_id_is_label:
                nm=class_id

            tc=train_count[nm]
            vc=val_count[nm]
            tec=test_count[nm]
            remc=remain_count[nm]
            if tec<test_nums[nm]:
                is_aug_file = False
                for word in test_exclusion_reg:
                    if str(nm).__contains__(word):
                        is_aug_file = True
                        break
                if not is_aug_file:
                    test_list.append(row)
                    tec += 1
                    test_count[nm] = tec
                else:
                    if tc <train_nums[nm]:
                        train_list.append(row)
                        tc += 1
                        train_count[nm] = tc
                    elif vc <val_nums[nm]:
                        val_list.append(row)
                        vc += 1
                        val_count[nm] = vc
                    else:
                        continue
            elif vc<val_nums[nm]:
                val_list.append(row)
                vc+=1
                val_count[nm]=vc

            elif  tc<train_nums[nm]:
                train_list.append(row)
                tc+=1
                train_count[nm]=tc
            else:#remain data
                remain_list.append(row)
                remc+=1
                remain_count[nm]=remc

        if len(train_list)>0:
            np.savetxt(train_csv,train_list,fmt="%s",delimiter=sep)
        if len(val_list)>0:
            np.savetxt(val_csv, val_list, fmt="%s", delimiter=sep)
        if len(test_list)>0:
            np.savetxt(test_csv, test_list, fmt="%s", delimiter=sep)
        if len(remain_list)>0:
            if remain_csv is not None:
                np.savetxt(remain_csv, remain_list, fmt="%s", delimiter=sep)
        print "saved %d train %d val, %d test, %d remain" %(len(train_list),len(val_list),len(test_list),len(remain_list))
        return


    def copy_images_between_datasets_by_regex(self,src_dataset,dst_dataset,img_ext=".bin.png",
                                     filename_sep="_",
                                              reg="RESIZED_[A-Za-z\-]+.bin.png"):
        iterate=0
        self.imitate_filetree(src_dataset,dst_dataset)
        p=re.compile(reg)
        for dirpath,dirs,files in os.walk(src_dataset):
            if len(files)>0:
                for file in files:
                    found=p.findall(file)
                    if len(found)>0:
                        src=os.path.join(dirpath,file)
                        tmpdir=dirpath[len(src_dataset)+1:]
                        dst=os.path.join(dst_dataset,tmpdir,file)
                        shutil.copyfile(src,dst)
                        print("COPIED:\r%s"%file)
        return

    def print_confusion_matrix_from_csv(self,csvf,
                                        sep=" ",
                                        gt_idx=0,
                                        predict_idx=-1,
                                        class_dict={}):
        df=pd.read_csv(csvf,sep=sep,header=None)
        npd=np.array(df)
        np_predict = npd[:,predict_idx]
        np_gt = npd[:,gt_idx]
        victorinox().print_confusion_matrix(ground_truths=np_gt,
                                            predictions=np_predict,
                                            class_dict=class_dict)
        correct_count = len([i for i, j in zip(np_predict, np_gt) if i == j])
        len_predict=len(np_predict)
        #print "ACC:%.5f"%(correct_count/len_predict)
        self.calculate_f1_measure(np_gt,np_predict)
        return

    def resize_and_recopy_image_less_than227x227(self,
                                                 srcdata,
                                                 destdata,
                                                 img_ext=".bin.png",
                                                 resize_dim=[227,227],
                                                 average_page_dim=[1776, 2546]):
        self.imitate_filetree(srcdata,destdata)
        aug=[]
        for dirpath,dirs,files in os.walk(srcdata):
            if len(files)>0:
                for fn in files:

                    try:
                        srcfp=os.path.join(dirpath,fn)
                        img=cv2.imread(srcfp)
                        h, w, c = img.shape
                        if h < resize_dim[1] or w < resize_dim[0]:
                            tmpd = dirpath[len(srcdata) + 1:]
                            clasname=str(fn).split("_")[-1].replace(img_ext,"")
                            fn2=str(fn).replace(clasname,"RESIZED_"+clasname)
                            destfp = os.path.join(destdata, tmpd, fn2)
                            image = Image.open(srcfp).convert("RGB")
                            zone_w, zone_h = image.size
                            resize_to = resize_dim
                            # splitted = str(fn).split("_")
                            # page_w = int(splitted[0])
                            # page_h = int(splitted[crossval1.old])
                            # x = int(splitted[2])
                            # y = int(splitted[3])
                            # if page_w == 0:
                            #     page_w = average_page_dim[0]
                            # if page_h == 0:
                            #     page_h = average_page_dim[crossval1.old]  # average height of images in dataset
                            #
                            # # image = Image.open(file_path).convert("RGB")  # RGB image process
                            # # if relocate:
                            # zone_w, zone_h = image.size
                            image.thumbnail(resize_to, Image.ANTIALIAS)
                            # # offset = (int((resize_to[0] - image.size[0]) / 2), int((resize_to[crossval1.old] - image.size[crossval1.old]) / 2))
                            page = Image.new("RGB", resize_to, "white")  # RGB image process
                            # # page = Image.new("crossval1.old", resize_to, "white")	# Black-white image process
                            #x1, y1, w1, h1 = self.map_coordinate_to_input_system(coord=[x, y],
                                                                                 # img_dim=[zone_w, zone_h],
                                                                                 # src_dim=[page_w, page_h],
                                                                                 # dest_dim=resize_dim)
                            page.paste(image,(0,0))
                            page.save(destfp)  # os.path.join(dest_dataset, basename(file)), ".png")
                            aug.append(destfp)
                            image.load()
                            page.load()
                            print "COPIED %s" % fn
                    except Exception,e:
                        print "ERROR:%s"%srcfp

        return aug

    def get_pixels_binarization_status_from_dataset(self,datasetpath="/a/b/c",sep=" ",sample_num=40):
        is_binarized=True
        found=[]
        count=0
        for dirpath,dirs,files in os.walk(datasetpath):
            if len(files)>0:
                for fn in files:
                    srcfp=os.path.join(dirpath,fn)
                    c = self.get_pixel_distribution_from_image_path(srcfp)
                    if len(c) > 2:  # if not binary
                        is_binarized = False
                        found.append(srcfp)
                    if count > sample_num:
                        break
                    count+=1

        return is_binarized,found

    def append_array_to_csv_file(self,arr,csvf,sep=" "):
        df=pd.read_csv(csvf,sep=sep,header=None)
        npd=np.array(df)
        npd2=np.array(arr)
        npd=np.concatenate(npd,npd2,1)
        np.savetxt(csvf,npd,fmt="%s",delimiter=" ")
        print "ADDED %d rows"%len(npd2)
        return

    def plot_train_val_test_distribution_per_dataset(self,
                                                     split_df={"train":[10,15,10,11,5],
                                                               "val": [5, 5, 5,5,5],
                                                               "test": [10, 15, 10,7,5]},
                                                     labels=["a","b","c","d"],
                                                     graph_label="UW III original balanced"):
        # import numpy as np
        # import matplotlib.pyplot as plt
        from matplotlib import rc
        # import pandas as pd

        # Data
        r = range(len(labels))#[0, crossval1.old, 2, 3, 4]
        raw_data = split_df#{'greenBars': [20, crossval1.old.5, 7, 10, 5], 'orangeBars': [5, 15, 5, 10, 15], 'blueBars': [2, 15, 18, 5, 10]}
        df = pd.DataFrame(raw_data)

        # From raw value to percentage
        totals = [i + j + k for i, j, k in zip(df['train'], df['val'], df['test'])]
        trains = [i for i in df["train"]]#[i / j * 100 for i, j in zip(df['train'], totals)]
        vals = [i for i in df["val"]]#[i / j * 100 for i, j in zip(df['val'], totals)]
        tests = [i for i in df["test"]]#[i / j * 100 for i, j in zip(df['test'], totals)]

        # plot
        barWidth = 0.85
        names = labels#('A', 'B', 'C', 'D', 'E')
        # Create green Bars
        plt.bar(r, trains, color='red', edgecolor='white', width=barWidth)
        # Create orange Bars
        plt.bar(r, vals, bottom=trains, color='green', edgecolor='white', width=barWidth)
        # Create blue Bars
        plt.bar(r, tests, bottom=[i + j for i, j in zip(trains, vals)], color='blue', edgecolor='white',
                width=barWidth)

        # Custom x axis
        plt.xticks(r, names)
        plt.xlabel(graph_label)
        red_patch = mpatches.Patch(color='red', label='train')
        green_patch = mpatches.Patch(color='green', label='val')
        blue_patch = mpatches.Patch(color='blue', label='test')
        plt.legend(handles=[red_patch,green_patch,blue_patch])
        plt.xticks(rotation=45)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel(graph_label)
        # Show graphic
        plt.show()
        return

    def calculate_f1_measure(self,GT=[],prediction=[]):
        accuracy=accuracy_score(GT, prediction)
        precision=precision_score(GT, prediction, average="macro")
        recall=recall_score(GT, prediction, average="macro")
        f1measure=f1_score(GT, prediction, average="macro")
        print "ACCURACY: %.5f"%accuracy
        print "PRECISION: %.5f"%precision
        print "RECALL: %.5f"%recall
        print "F1Measure: %.5f"%f1measure
        return (accuracy,precision,recall,f1measure)

    def show_table(self,dct={"col1":[],"col2":[]}):
        pt=PrettyTable()
        for k,v in dct.items():
            pt.add_column(k,v)
        print pt

    def show_thumb_grid(self, im_list=[], grid_shape=(4,4), scale=0.5, resizeto=[50,50],axes_pad=0.07):
        im_list=[np.array(cv2.imread(img)) for img in im_list]
        # Grid must be 2D:
        assert len(grid_shape) == 2

        # Make sure all images can fit in grid:
        assert np.prod(grid_shape) >= len(im_list)

        grid = ImageGrid(plt.gcf(), 111, grid_shape, axes_pad=axes_pad)
        N=len(im_list)
        for i in range(N):
            try:
                data_orig = im_list[i]

                # Scale image:
                im = PIL.Image.fromarray(data_orig)
                im=im.resize(resizeto)
                # thumb_shape = [int(scale * j) for j in im.size]
                # im.thumbnail(thumb_shape, PIL.Image.ANTIALIAS)
                #im.thumbnail(resizeto, PIL.Image.ANTIALIAS)
                data_thumb = np.array(im)
                grid[i].imshow(data_thumb)
                # Turn off axes:
                grid[i].axes.get_xaxis().set_visible(False)
                grid[i].axes.get_yaxis().set_visible(False)
            except Exception,e:
                print "ERROR:%s"%e
                continue
        plt.show()


    def plot_model_accloss(self,loss_path=["/a/b.csv"],acc_path=["/a/b.csv"],csv_sep=","):
        accdf=pd.read_csv(acc_path,sep=csv_sep,header=None)
        accarr=np.array(accdf)
        acc_epoch=range(1,len(accarr)+1)
        acc_train=accarr[:,1]
        acc_val = accarr[:, -1]

        lossdf = pd.read_csv(loss_path, sep=csv_sep,header=None)
        lossarr = np.array(lossdf)
        loss_epoch = range(1, len(lossarr) + 1)
        loss_train = lossarr[:, 1]
        loss_val = lossarr[:, -1]
        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(loss_epoch,loss_train,"r",loss_epoch,loss_val,"g")
        ax[0].set_title("LOSS")
        #ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].legend(loc="upper right")
        ax[1].plot(acc_epoch, acc_train, "r", acc_epoch, acc_val, "g")
        ax[1].set_title("ACCURACY")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        ax[1].legend(loc="upper right")

        plt.show()


    def plot_acc_and_f1_of_models(self,result_csvs=["/a.csv"],model_names=["ocrd_ori"],csv_sep=" ",gt_idx=0,predict_idx=-1):
        acc_f1_arr=[]
        for i in range(len(result_csvs)):
            result_csv=result_csvs[i]
            model=model_names[i]
            df0=pd.read_csv(result_csv,sep=csv_sep,header=None)
            npd0=np.array(df0)
            gt=npd0[:,gt_idx]
            pred=npd0[:,predict_idx]
            (acc,prec,rec,f1)=self.calculate_f1_measure(gt,pred)
            acc_f1_arr.append(["acc",model,acc])
            acc_f1_arr.append(["f1", model, f1])
        # df = pd.DataFrame([['g1', 'c1', 10], ['g1', 'c2', 12], ['g1', 'c3', 13], ['g2', 'c1', 8],
        #                    ['g2', 'c2', 10], ['g2', 'c3', 12]], columns=['group', 'column', 'val'])
        df=pd.DataFrame(acc_f1_arr,columns=['evaluation', 'model', 'score'])
        df.pivot("model", "evaluation", "score").plot(kind='bar')
        plt.ylabel("score")

        plt.xticks(rotation=20)
        plt.show()
        return

    def get_average_of_JensenShannon_using_tensorflow(self, histA, histB):
        histogram_types_num=12
        distances = 0
        arr_a=np.array(histA)
        arr_b = np.array(histB)
        hbx_a=tf.Variable(arr_a[0:8],dtype=tf.float32,trainable=False,name="hbx_a",)
        hwx_a = tf.Variable(arr_a[8:16],dtype=tf.float32,trainable=False,name="hwx_a")
        hbwx_a = tf.Variable(arr_a[16:24],dtype=tf.float32,trainable=False,name="hbwx_a")
        hby_a = tf.Variable(arr_a[24:32],dtype=tf.float32,trainable=False,name="hby_a")
        hwy_a = tf.Variable(arr_a[32:40],dtype=tf.float32,trainable=False,name="hwy_a")
        hbwy_a = tf.Variable(arr_a[40:48],dtype=tf.float32,trainable=False,name="hbwy_a")

        hbx_b = tf.Variable(arr_b[0:8],dtype=tf.float32,trainable=False,name="hbx_b")
        hwx_b = tf.Variable(arr_b[8:16],dtype=tf.float32,trainable=False,name="hwx_b")
        hbwx_b = tf.Variable(arr_b[16:24],dtype=tf.float32,trainable=False,name="hbwx_b")
        hby_b = tf.Variable(arr_b[24:32],dtype=tf.float32,trainable=False,name="hby_b")
        hwy_b = tf.Variable(arr_b[32:40],dtype=tf.float32,trainable=False,name="hwy_b")
        hbwy_b = tf.Variable(arr_b[40:48],dtype=tf.float32,trainable=False,name="hbwy_b")

        hbx_d = self.getJensenShannonDistance_using_tensorflow(hbx_a,hbx_b)
        hwx_d = self.getJensenShannonDistance_using_tensorflow(hwx_a, hwx_b)
        hbwx_d = self.getJensenShannonDistance_using_tensorflow(hbwx_a, hbwx_b)
        hby_d = self.getJensenShannonDistance_using_tensorflow(hby_a, hby_b)
        hwy_d= self.getJensenShannonDistance_using_tensorflow(hwy_a, hwy_b)
        hbwy_d = self.getJensenShannonDistance_using_tensorflow(hbwy_a, hbwy_b)

        #mean_d=1.0*np.mean([hbx_d,hwx_d,hbwx_d,hby_d,hwy_d,hbwy_d])
        arr_result=tf.Variable([hbx_d,hwx_d,hbwx_d,hby_d,hwy_d,hbwy_d],trainable=False,dtype=tf.float32)
        mean_d=tf.reduce_mean(arr_result)
        return mean_d

    def getJensenShannonDistance_using_tensorflow(self, P, Q):
        # print "test-white:%s--train_white:%s" %(str(P),str(Q))
        half=tf.Constant(0.5,dtype=tf.float32)
        _P = tf.div(P,tf.norm(P,ord=1))#P / norm(P, ord=1)
        _Q = tf.div(Q,tf.norm(Q,ord=1))#Q / norm(Q, ord=1)
        _M = tf.multiply (half, (tf.add(_P,_Q)))

        ent1=tf.add_n(tf.multiply(_P,(tf.log(tf.div(_P,_M)))))
        ent2 = tf.add_n(tf.multiply(_Q, (tf.log(tf.div(_Q, _M)))))
        ent_add=tf.add(ent1,ent2)
        ret=tf.multiply(half,ent_add)
        return ret#0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    def classify_by_knn_tensorflow(self,X_t, y_t, x_t, k_t):
        neg_one = tf.constant(-1.0, dtype=tf.float64)
        # we compute the L-1 distance
        #distances = tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
        distances = self.get_average_of_JensenShannon_using_tensorflow(X_t,x_t)
        # to find the nearest points, we find the farthest points based on negative distances
        # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
        neg_distances = tf.multiply(distances, neg_one)
        # get the indices
        vals, indx = tf.nn.top_k(neg_distances, k_t)
        # slice the labels of these points
        y_s = tf.gather(y_t, indx)
        return y_s
        #return

    def get_label(preds):
        counts = np.bincount(preds.astype('int64'))
        return np.argmax(counts)

    def find_string_in_dataset_csv(self,dataset="/a.csv",sep=",",reg="(SHEAR|ZOOMED)"):
        df=pd.read_csv(dataset,sep=sep,header=None)
        npd=np.array(df)
        p=re.compile(reg)
        has_string=False
        for row in npd:
            found=p.findall(row[0])
            if len(found)>0:
                #print row
                has_string=True
                break
        return has_string

    def convert_label_to_id(self,csvfile="a.csv",csvtarget="b.csv",sep=",",dct={1:"a",2:"b"}):
        df=pd.read_csv(csvfile,sep=sep,header=None)
        npd=np.array(df)
        arr=[]
        revdct={v:k for k,v in dct.items()}
        for row in npd:
            arr.append([revdct[row[0].strip()],revdct[row[1].strip()]])
        np.savetxt(csvtarget,arr,fmt="%s",delimiter=",")
        print "SAVED %d rows to %s"%(len(arr),csvtarget)
































