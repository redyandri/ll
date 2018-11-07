from __future__ import division

import sys
import csv
from RLXYMSH import RLXYMSH
import os
import re
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
from shutil import copytree

class my_tool(object):
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

    def select_hifreq_class(self,class_to_omit,csv_src,csv_dest):
        p = csv_src
        df = pd.read_csv(p)
        minor = class_to_omit
            # [
            # "list-item",
            # "reference-list_item"
            # "pseudo-code",
            # "logo",
            # "synopsis",
            # "advertisement",
            # "map",
            # "publication-info",
            # "highlight",
            # "reader-service",
            # "keyword_heading_and_body",
            # "seal",
            # "membership",
            # "diploma",
            # "correspondence",
            # "announcement",
            # "abstract_heading_and_body"]
        high_freq = np.array([row for row in df.itertuples(index=False) if row[-1] not in minor])  #skip rows whose class is minor or to omit
        with open(csv_dest, "wb") as dest:
            writer = csv.writer(dest)
            for row in high_freq[:]:
                row = row.tolist()
                writer.writerow(row)
            dest.close()

    # def print_confusion_matrix(self,ground_truths,predictions):
    #     set1 = np.vstack((predictions, ground_truths))
    #     class_set = np.unique(set1) #get set of labels
    #     sorted_set = sorted(class_set)
    #     sorted_set = np.array(sorted_set)
    #     cm = confusion_matrix(ground_truths.tolist(), predictions.tolist())
    #     h_labels = np.reshape(sorted_set, (1, sorted_set.shape[0]))
    #     cm = np.vstack((h_labels, cm))
    #     added_sort = np.concatenate((["*"], sorted_set))
    #     v_labels = np.reshape(added_sort, (added_sort.shape[0], 1))
    #     cm = np.hstack((v_labels, cm))
    #     cols = [i[0] for i in enumerate((cm[0, :]))]
    #     pt = PrettyTable(field_names=cols)
    #     for row in cm[1:, :]:
    #         pt.add_row(row.tolist())
    #     print pt
    #     for i in cols:
    #         if i>0:
    #             print "%d : %s" %(i,cm[0,i])
    def print_confusion_matrix(self,ground_truths,predictions,class_dict={}):
        #predictions=[class_dict[p] for p in predictions]
        #ground_truths = [class_dict[g] for g in ground_truths]
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
            i=1
            for r in row[1:]:
                if total==0:
                    row[i]=0
                else:
                    row[i]=round(int(r)/total,2)
                i+=1
            pt.add_row(row.tolist())
        print pt
        for i in cols:
            if i>0:
                print "%d : %s" %(i,cm[0,i])

    def get_classname_dictionary(self,dataset_path,img_ext=".bin.png",filename_sep="_",classname_idx=-1):
        class_dist=self.get_classname_distribution(dataset_path=dataset_path,img_ext=img_ext,name_splitter=filename_sep,start_classname_idx=classname_idx)
        classname_enum = list(enumerate([x for (x, y) in class_dist]))
        class_dictionary = {id:classname for id, classname in classname_enum}
        return class_dictionary

    def get_classname_distribution(self,dataset_path,img_ext=".bin.png",name_splitter="_",start_classname_idx=-1,end_classname_idx=0):
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
                            class_coord = [img_f, class_type, coords.split(" ")]
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



    def imitate_image_filetree(self,src_dataset,dest_dataset,):
        for dirpath, dirnames, filenames in os.walk(src_dataset):
            if not dirpath.endswith("page"): #select only jpg folder containg images
                structure = os.path.join(dest_dataset, dirpath[len(src_dataset)+1:])
                if not os.path.exists(structure):
                    os.mkdir(structure,0777)
                    print "Created: %s" %str(structure)

    def find_image_path(self,name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)


    def get_class_collection(self,dataset_path,img_ext=".TIF",sep="_",class_idx=-1):
        tool = my_tool()
        classes = []
        for dirpath, dirs, files in os.walk(dataset_path):
            if len(files)>0:
                for crop in files:
                    if str(crop).endswith(img_ext):
                        crop_path = os.path.join(dirpath, crop)
                        class_type = str(crop).split(sep)[class_idx]
                        class_type.replace(img_ext,"")
                        classes.append(class_type)
        c = Counter(classes)
        c = c.most_common()
        print c

    def generate_mini_dataset(src_dataset_path, mini_dataset_path, txt_path):
        tool = my_tool()
        classes = []
        num_per_class = 137  # number of endnote=137, the least numbered class
        class_dict = {
            'paragraph': 0,
            'page-number': 0,
            'catch-word': 0,
            'header': 0,
            'heading': 0,
            'signature-mark': 0,
            'other': 0,
            'Separator': 0,
            'footnote': 0,
            'marginalia': 0,
            'Graphic': 0,
            'Maths': 0,
            'caption': 0,
            'Table': 0,
            'footnote-continued': 0,
            'endnote': 0
        }
        for dirpath, dirs, files in os.walk(src_dataset_path):
            if dirpath.endswith("jpg"):
                for crop in files:
                    crop_path = os.path.join(dirpath, crop)
                    class_type = str(crop).split("_")[-2]
                    if class_type == "":
                        continue
                    class_num = class_dict[class_type]
                    if class_num <= num_per_class:
                        row = [crop_path, class_type]
                        classes.append(row)
                        class_num += 1
                        class_dict[class_type] = class_num
                    else:
                        continue
        with open(txt_path, "w") as output:
            for c in classes:
                src_file = c[0]
                head, tail = os.path.split(src_file)
                dest_file = os.path.join(mini_dataset_path, tail)
                class_type = c[1]
                shutil.copyfile(src_file, dest_file)
                row = dest_file + " " + class_type
                output.write(row)
                output.write("\n")
                print "COPIED : %s" % tail
            output.flush()
            output.close()
        print "%d CLASSES COPIED." %len(classes)
        return classes

    def delete_files_in_folder(self,folder_path,file_extension):
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

    def split_train_test(self,dataset_txt_path,train_txt_path,test_txt_path):
        df=pd.read_csv(dataset_txt_path,sep=" ",header=None)
        data=np.array(df.ix[:,:])
        image_paths=np.array(data[:,0])
        class_labels=np.array(data[:,1])
        trains,tests=train_test_split(data)
        np.savetxt(train_txt_path,trains,delimiter=" ",fmt="%s %s")
        np.savetxt(test_txt_path, tests, delimiter=" ",fmt="%s %s")
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

    def generate_highfreq_class_dataset(self, src_dataset_path="a/b/c",dest_dataset_path="a/b/c",highfreq_class=["a","b"],img_ext=".TIF",sep="_"):
        low_freq=[]
        c=0
        self.imitate_image_filetree(src_dataset_path,dest_dataset_path)
        for dirpath,dirs,files in os.walk(src_dataset_path):
            if len(files)>0:
                for file in files:
                    if str(file).endswith(img_ext):
                        spli=str(file).split(sep)
                        class_name=spli[-1]
                        class_name=class_name[0:str(file).index(img_ext)]
                        if class_name in highfreq_class:
                            src_fp=os.path.join(dirpath,file)
                            dst_fp=dirpath[len(src_dataset_path)+1:]
                            dst_fp=os.path.join(dest_dataset_path,dst_fp,file)
                            shutil.copyfile(src_fp,dst_fp)
                            print "copied %s" %file
                            c+=1
                        else:
                            low_freq.append(file)
                            continue
        print "copied %d high freq files" %c
        print "ignored %d low freq files" % len(low_freq)








