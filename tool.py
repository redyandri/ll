from __future__ import division
import sys
import math
# sys.path.insert(0,"../")
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import cv2
#from __future__ import division
from numpy import median
from time import time
from tool.csv_feature_generator import *
from KNNClassifier import KNNClassifier
from my_tool import *
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import pandas as pd
from PIL import Image
from tensorflow_knn import tf_knn
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
import xml.etree.ElementTree as ET
from collections import Counter
import csv
import shutil
from victorinox import victorinox

import tensorflow as tf
import os
import matplotlib.pyplot as plt


tool=victorinox()
#tool.reshape_and_relocate_zones(ocrd_hifreq,ocrd_reshaped_relocated_blackpadded_224)
#.convert_dataset_to_csv(ocrd_reshaped_relocated_blackpadded_224,ocrd_reshaped_relocated_blackpadded_224_csv)
# tool.remove_black_pixels_majored_images_from_csv(ocrd_reshaped_relocated_blackpadded_224_csv,ocrd_reshaped_relocated_blackpadded_unblacked_224_csv)
# tool.generate_mini_dataset_from_csv(ocrd_reshaped_unrelocated_whitepadded_224_csv,ocrd_reshaped_unrelocated_whitepadded_224_csv)
src="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/224/blackpadded/dataset.csv"
dst="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/224/whitepadded/dataset.csv"
train1="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/227/blackpadded/holdout/train.csv"
val1="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/227/blackpadded/holdout/val.csv"
test1="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/227/blackpadded/holdout/test.csv"
mini="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/mini/relocated/224/blackpadded/mini.csv"
img="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/relocated/224/blackpadded/praetorius_syntagma01_1615_zkg/jpg/2000_2846_1623_2379_clip_praetorius_syntagma01_1615_0496_catch-word.bin.png"
balanced_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/227/blackpadded/balanced.csv"
remain_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/227/blackpadded/remain.csv"
train_mini="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/227/blackpadded/mini/train.csv"
val_mini="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/227/blackpadded/mini/val.csv"
test_mini="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/relocated/227/blackpadded/mini/test.csv"
# print tool.get_classname_distribution_from_csv(csv_path=train1,sep =" ")
# print tool.get_classname_distribution_from_csv(csv_path=val1,sep =" ")
# print tool.get_classname_distribution_from_csv(csv_path=test1,sep =" ")
#tool.generate_balanced_train_val_by_quantity(dataset_csv=balanced_csv,train_csv=train_mini,val_csv=val_mini,test_csv=test_mini,train_num=100,val_num=100)
hifreq="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq"
unreloc="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227"
unreloc256="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/256"
unreloc_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/dataset.csv"
balanced_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/balanced.csv"
remain_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/remain.csv"
remainnoaugment_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/remain_noaugment.csv"
train_mini="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/train.csv"
val_mini="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/val.csv"
test_mini="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/test.csv"
train2="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/train.csv"
val2="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/val.csv"
test2="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/test.csv"
test2_noaugment="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/test_noaugmentation.csv"
unreloc256_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/256/dataset.csv"
#tool.reshape_and_relocate_zones(src_dataset=hifreq,dest_dataset=unreloc256,resize_dim=[256,256],relocate=False)
#tool.convert_dataset_to_csv(dataset_path=unreloc256,csv_path=unreloc256_csv)
#tool.augment_class_from_csv(dataset_csv=unreloc256_csv)
#tool.convert_dataset_to_csv(dataset_path=unreloc,csv_path=unreloc_csv)
#tool.binarize_dataset_with_otsu(src_csv=unreloc_csv)
#print tool.get_classname_distribution_from_csv(csv_path=unreloc_csv,sep=" ")
#tool.generate_balanced_dataset(dataset_csv=unreloc_csv,target_balanced_csv=balanced_csv,target_remain_csv=remain_csv,num_per_class=1000)
#tool.generate_balanced_train_val_by_quantity(dataset_csv=balanced_csv,train_csv=train_mini,val_csv=val_mini,test_csv=test_mini,train_num=100,val_num=100)
#print tool.get_pixels_binarization_status_from_dataset_csv(unreloc_csv,sample_num=100)
#tool.generate_balanced_train_val_by_quantity(dataset_csv=balanced_csv,train_csv=train2,val_csv=val2,test_csv=test2,train_num=800,val_num=100)
#tool.remove_dataset_images_by_regex_from_csv(dataset_csv=remain_csv,target_csv=remainnoaugment_csv,reg="_AUGMENT_")
# print tool.get_classname_distribution_from_csv(csv_path=remain_csv,sep =" ")
# print tool.get_classname_distribution_from_csv(csv_path=remainnoaugment_csv,sep =" ")
#print tool.get_classname_distribution_from_csv(csv_path=test2,sep =" ")

# for row in np1:
#     fp=row[0]
#     if str(fp).__contains__("_AUGMENT_"):
#         continue
#     else:
#         combine.append(row)
# for row in np2:
#     fp=row[0]
#     if str(fp).__contains__("_AUGMENT_"):
#         continue
#     else:
#         combine.append(row)
# np.savetxt(remainnoaugment_csv,combine,fmt="%s",delimiter=" ")
# print "REMAIN:%s"%tool.get_classname_distribution_from_csv(csv_path=remain_csv,sep=" ")
# print "REMAIN_NOAUG:%s"%tool.get_classname_distribution_from_csv(csv_path=remainnoaugment_csv,sep=" ")
# print "TEST:%s"%tool.get_classname_distribution_from_csv(csv_path=test2,sep=" ")
# #tool.remove_dataset_images_by_regex_from_csv(dataset_csv=test2,target_csv=test2_noaugment,reg="_AUGMENT_")
# print "TEST NOAUG:%s"%tool.get_classname_distribution_from_csv(csv_path=test2_noaugment,sep=" ")
# csv_arr=[test2_noaugment,remainnoaugment_csv]
# combine=[]
# for csvs in csv_arr:
#     df1=pd.read_csv(csvs,sep=" ")
#     np1=np.array(df1)
#     for row in np1:
#         combine.append(row)
# np.savetxt(remainnoaugment_csv,combine,fmt="%s",delimiter=" ")
#
# print "REMAIN_NOAUG NOW:%s"%tool.get_classname_distribution_from_csv(csv_path=remainnoaugment_csv,sep=" ")
uw3="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/zones/uw3_zones_noredundant_hifreq"
uw3_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/zones/hifreq/dataset.csv"
ocrd="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq"
ocrd_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/zones/hifreq/dataset.csv"
unreloc227_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/dataset.csv"
unreloc227="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/"
import Augmentor
#tool.convert_dataset_to_csv(dataset_path=unreloc227,csv_path=unreloc227_csv)
#print tool.get_classname_distribution("/home/andri/Pictures/jpg/")
#tool.remove_dataset_images_by_regex_from_csv(dataset_csv=unreloc227_csv,target_csv=unreloc227_csv,reg="_AUGMENT_")




# tool.convert_dataset_to_csv("/home/andri/Pictures/abel_leibmedicus_1699_zkg_ori",csv_path="/home/andri/Pictures/abel_leibmedicus_1699_zkg_ori.csv")
# print tool.get_pixels_binarization_status_from_dataset_csv("/home/andri/Pictures/abel_leibmedicus_1699_zkg_ori.csv")
# # print tool.get_classname_dictionary("/home/andri/Pictures/abel_leibmedicus_1699_zkg/")
# # print tool.get_classname_distribution(dataset_path="/home/andri/Pictures/jpg/")
# tool.augment_class2(dataset_path="/home/andri/Pictures/abel_leibmedicus_1699_zkg_ori/",class_id=9,augment_num=1)
#print tool.get_pixels_binarization_status_from_dataset_csv("/home/andri/Pictures/abel_leibmedicus_1699_zkg_ori.csv")
# tool.convert_dataset_to_csv(dataset_path=uw3,csv_path=uw3_csv)
# tool.convert_dataset_to_csv(dataset_path=ocrd,csv_path=ocrd_csv)
# dist1=tool.get_classname_distribution_from_csv(uw3_csv,sep=" ")
# dist2=tool.get_classname_distribution_from_csv(ocrd_csv,sep=" ")
# print dist1
# print
ori = "/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq"
shp227 = "/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227"
shp227csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/dataset.csv"
shp227balanced="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/balanced.csv"
remain_csv2="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/remain.csv"
remainnoaugment_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/remain_noaugment.csv"

train="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/train.csv"
val="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/val.csv"
test="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/test.csv"
#tool.generate_balanced_dataset(dataset_csv=shp227csv,target_balanced_csv=shp227balanced,target_remain_csv=shp227remain,num_per_class=1300)
# tool.generate_balanced_train_val_by_quantity(dataset_csv=shp227balanced,train_csv=train,val_csv=val,test_csv=test,train_num=1200,val_num=100)
# #print tool.get_classname_dictionary(dataset_path=ori)
# #print tool.get_classname_dictionary(dataset_path=shp227)
# #tool.convert_dataset_to_csv(dataset_path="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq",csv_path="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq.csv")
# print tool.get_classname_distribution_from_csv(csv_path=shp227csv,sep=" ")
# print tool.get_classname_distribution_from_csv(csv_path=shp227balanced,sep=" ")
# print tool.get_classname_distribution_from_csv(csv_path=shp227remain,sep=" ")
# print tool.get_classname_distribution_from_csv(csv_path=train,sep=" ")
# print tool.get_classname_distribution_from_csv(csv_path=val,sep=" ")
# print tool.get_classname_distribution_from_csv(csv_path=test,sep=" ")

#tool.augment_class2(dataset_path=ori,target_path=shp227,class_id=11,augment_num=2)
#
# dataset_all="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/zones/uw3_zones_noredundant_allclass"
# dataset_path="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/zones/uw3_zones_noredundant_hifreq"
# target_path="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented"
# target_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/dataset.csv"
# balanced_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/balanced.csv"
# remain_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/remain.csv"
# remain_no_augmentation_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/remain_no_augmentation.csv"
# balanced_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/holdout/train.csv"
# balanced_val_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/holdout/val.csv"
# balanced_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/holdout/test.csv""/a/b/c.csv"
# balanced_train_num=1200
# balanced_val_num=100
# augmented_num_per_class=1500
# balanced_class_num=1300
# mini_balanced_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/mini/balanced.csv"
# mini_remained_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/mini/remain.csv"
# mini_remained_no_augmentation_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/mini/remain_no_augmentation.csv"
# mini_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/mini/holdout/train.csv"
# mini_val_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/mini/holdout/val.csv"
# mini_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/mini/holdout/test.csv"
# mini_train_num=100
# mini_val_num=100
# mini_balanced_per_class_num=300
# crossval_path="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/crossval"
# mini_crossval_path="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/mini/crossval"
# img_ext=".bin.png"
# name_splitter="_"
dataset_all="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/allclass"
dataset_path="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq_topleftcrop227"
target_path="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented"
target_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/augmented.csv"
balanced_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/balanced.csv"
remain_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/remain.csv"
remain_no_augmentation_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/remain_no_augmentation.csv"
balanced_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/train.csv"
balanced_val_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/val.csv"
balanced_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/holdout/test.csv"
balanced_train_num=1200
balanced_val_num=100
augmented_num_per_class=1500
balanced_class_num=1300
mini_balanced_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/balanced.csv"
mini_remained_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/remain.csv"
mini_remained_no_augmentation_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/remain_no_augmentation.csv"
mini_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/train.csv"
mini_val_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/val.csv"
mini_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/test.csv"
mini_train_num=200
mini_val_num=200
mini_balanced_per_class_num=500
crossval_path="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/crossval"
mini_crossval_path="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/crossval"
img_ext=".bin.png"
name_splitter="_"
zones_leftcrop227="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq_topleftcrop227"

# p="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/unaugmented"
# tool.remove_dataset_images_by_regex(dataset=p,img_ext=".bin.png",reg="_AUG")



# print tool.get_classname_dictionary(dataset_path=dataset_all)
# print tool.get_classname_distribution(dataset_path=dataset_all)
# print tool.get_classname_dictionary(dataset_path=shp227)
# print tool.get_classname_distribution(dataset_path=shp227)
#tool.show_cyclical_learning_rate_flow(num_epochs=400,train_num=2100)
from matplotlib import pyplot as plt
# f=open("temp/test.csv","a")
# writer=csv.writer(f)
# for i in range(10):
#     writer.writerow([i,50+i,100+i])
# f.close()
#
# df=pd.read_csv("temp/test.csv",sep=",")
# npd=np.array(df)
# fig,ax=plt.subplots()
# ax.plot(npd[:,0],npd[:,1],"g",label="val")
# ax.plot(npd[:,0],npd[:,2],"r",label="train")
# leg=ax.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
# plt.show()
s="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented"
train="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped/train"
val="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped/val"


uw3="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_csv.csv"
ocrd="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_csv.csv"
uw3_clustered="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_CLUSTERED_csv.csv"
ocrd_clustered="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_CLUSTERED_csv.csv"
uw3_clustered_balanced="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_CLUSTERED_BALANCED_csv.csv"
ocrd_clustered_balanced="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_CLUSTERED_BALANCEDcsv.csv"

uw3_clusters=["abstract-body_caption_footnote_list-item_text-body","author_page-header_title_section-heading"]
ocrd_clusters=["endnote_footnote-continued_paragraph"]

# tool.rename_to_similarity_group_from_csv(uw3,uw3_clustered,",",similarity_group=uw3_clusters)
# tool.rename_to_similarity_group_from_csv(ocrd,ocrd_clustered,",",ocrd_clusters)
#tool.generate_balanced_dataset(uw3_clustered,uw3_clustered_balanced,sep=",")
#print tool.get_classname_distribution_from_csv(uw3_clustered_balanced,",")
l="/home/andri/Documents/s2/5/master_arbeit/documentation/deep learning/ocrd/unrelocated/227/mini/myalexnet/200/losses.csv"
a="/home/andri/Documents/s2/5/master_arbeit/documentation/deep learning/ocrd/unrelocated/227/mini/myalexnet/200/accuracies.csv"
#tool.plot_trainval_accloss(loss_path=l,acc_path=a,csv_sep=",")
temp="/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/temp"
zones="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq"
zones_topleftcrop227="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq_topleftcrop227"
unrecolated227="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/unaugmented"
# dct=tool.get_classname_dictionary(zones)
# print dct
# print tool.get_classname_distribution(zones)
# ids=[15,14,8]
#for k,v in dct.items():
#tool.augment_class_by_leftcrop(dataset_path=zones,target_path=zones_topleftcrop227)
target_path="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented/augmented_confused_class"
# tool.prepare_dataset(src_dataset_path=zones,
#                      target_augmented_path=target_path,
#                      target_augmented_csv=target_csv,
#                      balanced_csv=balanced_csv,
#                      remain_csv=remain_csv,
#                      remain_no_augmentation_csv=remain_no_augmentation_csv,
#                      balanced_train_csv=balanced_train_csv,
#                      balanced_val_csv=balanced_val_csv,
#                      balanced_test_csv=balanced_test_csv,
#                      balanced_train_num=balanced_train_num,
#                      balanced_val_num=balanced_val_num,
#                      augmented_num_per_class=augmented_num_per_class,
#                      balanced_class_num=balanced_class_num,
#                      mini_balanced_csv=mini_balanced_csv,
#                      mini_remained_csv=mini_remained_csv,
#                      mini_remained_no_augmentation_csv=mini_remained_no_augmentation_csv,
#                      mini_train_csv=mini_train_csv,
#                      mini_val_csv=mini_val_csv,
#                      mini_test_csv=mini_test_csv,
#                      mini_train_num=mini_train_num,
#                      mini_val_num=mini_val_num,
#                      mini_balanced_per_class_num=mini_balanced_per_class_num,
#                      crossval_path=crossval_path,
#                      mini_crossval_path=mini_crossval_path,
#                      img_ext=img_ext,
#                      name_splitter=name_splitter)
augmented="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented/augmented_confused_class"
mini="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/mini"
minidatasetcsv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/dataset.csv"
minitraincsv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/train.csv"
minivalcsv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/val.csv"
minitestcsv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini/test.csv"
miniar="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/mini_ar"
miniardatasetcsv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini_ar/dataset.csv"
miniartraincsv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini_ar/train.csv"
miniarvalcsv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini_ar/val.csv"
miniartestcsv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/mini_ar/test.csv"
# tool.prepare_mini_dataset(src_dataset_path=zones,
#                           mini_dataset_path=mini,mini_dataset_csv=miniardatasetcsv,
#                           mini_train_csv=miniartraincsv,
#                           mini_val_csv=miniarvalcsv,
#                           mini_test_csv=miniartestcsv
#                           )

grouped2="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped2"
endnote="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped/endnote"
footnote="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped/footnote"
footnote_continued="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped/footnote-continued"
paragraph="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped/paragraph"
class_to_topleftcrop=["endnote","footnote","footnote-continued","paragraph"]
# print tool.get_convolvable_dim(227,227,kernels=[11,3,3,3],strides=[4,2,2,2])
# print tool.get_convolvable_dim(291,291,kernels=[11,3,3,3],strides=[4,2,2,2])


#tool.group_images_per_classname(dataset_path=zones,target_path=grouped2,num_per_class=100)
#tool.generate_dataset_by_topleftcrop(grouped2,grouped2,crop_dim=[291,291],classname_to_crop=class_to_topleftcrop)
#print tool.get_classname_distribution(augmented)

train="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped224/trainval_grouped/train"
val="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped224/trainval_grouped/val"
test="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped224/trainval_grouped/test"

#tool.generate_balanced_train_val_by_quantity2(grouped2,train,val,test,train_num=0,val_num=0,test_num=100)
# tenso=tf.constant([[[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]],
#                    [[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]],dtype=tf.float32)
#
# # zeroes=tf.zeros([tenso.get_shape()[0],tenso.get_shape()[1],tenso.get_shape()[2],5])
# # concat=tf.concat([tenso,zeroes],axis=3)
# def zeropad_to_increase_channel(x, output_channel=256):
#     zeroes = tf.zeros([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], output_channel-tf.shape(x)[-1]])
#     concat = tf.concat([x, zeroes], axis=3)
#     return concat
# with tf.Session() as sess:
#     concat=zeropad_to_increase_channel(tenso,output_channel=100)
#     print sess.run(tf.shape(tenso))
#     print sess.run(tf.shape(concat))
#     print sess.run(concat)

#print tool.get_branch_channel(fold=6,out_channel=96)
p="/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/saved_sessions/checkpoints/ocrd/unrelocated/227/mini/myalexnet/sgd/clr/1/saved_model/accuracies.csv"

# dat=pd.read_csv(p,sep=",")
# npd=np.array(dat)
# print npd[-10:,-1]
# print np.mean(npd[-10:,-1])
# print np.std(npd[-10:,-1])
groupedocrd="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/grouped"
mini_data="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/mini/dataset"
mini_train="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped224/mini/trainvaltest/train"
mini_val="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped224/mini/trainvaltest/val"
mini_test="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped224/mini/trainvaltest/test"
mini_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/grouped224/mini/train.csv"
mini_val_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/grouped224/mini/val.csv"
mini_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/grouped224/mini/test.csv"
mini_data="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/grouped224/mini/dataset"
ocrdhifreq="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/hifreq"
ocrdallclass="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/zones/allclass"
uw3zones="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/zones/uw3_zones_noredundant_hifreq"
class_to_group=["endnote"]
# tool.group_images_per_classname(zones,target_path=groupedocrd,num_per_class=1500,resize_dim=[227,227])
# tool.add_topleftcrop_to_dataset(groupedocrd,target_path=groupedocrd,crop_dim=[227,227],classname_to_crop=class_to_topleftcrop)
# tool.generate_balanced_train_val_by_quantity2(dataset_path=grouped224,train_path=mini_train,val_path=mini_val,test_path=mini_test,train_num=100,val_num=100,test_num=100,resize_dim=[224,224],val_and_test_exclude_keword="CROP")
# tool.convert_dataset_to_csv(dataset_path=mini_train,csv_path=mini_train_csv)11
# tool.convert_dataset_to_csv(dataset_path=mini_val,csv_path=mini_val_csv)
# tool.convert_dataset_to_csv(dataset_path=mini_test,csv_path=mini_test_csv)
#print tool.get_classname_distribution_from_csv(csv_path="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/unrelocated/227/remain.csv",sep=" ")
#print tool.get_convolvable_dim(h=227,w=227,kernels=[11,3,3,3],strides=[4,2,2,2])
# print tool.get_pixels_binarization_status_from_dataset_csv(minidatasetcsv)
# tool.binarize_dataset_with_otsu(minidatasetcsv)
# print tool.get_pixels_binarization_status_from_dataset_csv(minidatasetcsv)

src="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_csv.csv"
bal="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/balanced.csv"
rem="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/remain.csv"
cross="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/crossval"
# print tool.get_classname_distribution_from_csv(src,sep=",")
#
# tool.generate_balanced_dataset(src,bal,rem,",")
# print tool.get_classname_distribution_from_csv(bal,sep=",")
# tool.generate_crossvalidation_from_csv(bal,sep=",",fold_num=10,target_folder=cross)
# print tool.get_classname_distribution_from_csv(bal,sep=",")
#print tool.get_classname_distribution_from_csv("/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/crossval/balanced_train_10.csv",sep=",")
#print tool.get_classname_distribution_from_csv("/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/crossval/balanced_val_10.csv",sep=",")
# dist=tool.get_classname_distribution(ocrdallclass)
# num=len(dist)
# print num
# print dist
ocrdhist="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_csv.csv"
ocrdallclasshist="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_allclass_histogram_csv.csv"
ocrdorihist="/home/andri/Documents/s2/5/master_arbeit/app/logical_labeling/dataset/histograms/ocrd_orisize_rlbwxyh_16classes.csv"
uw3allclasshist="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_csv.csv"
uw3="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_csv.csv"
ocrd="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_csv.csv"
uw3allfolder="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/zones/uw3_zones_noredundant_allclass"
# dist=tool.get_classname_distribution_from_csv(ocrdallclasshist,sep=",")
# #num=sum([y for x,y in dist])
# print len(dist)
# dist=tool.get_classname_distribution_from_csv(ocrdhist,sep=",")
# num=sum([y for x,y in dist])
# print num
# dist=tool.get_classname_distribution_from_csv(uw3,sep=",")
# num=sum([y for x,y in dist])
# print num
# #print "OCRD-->TOTAL:%d,TRAIN:%d,TEST:%d"%(num,0.8*num,0.2*num)
# print dist
# dist=tool.get_classname_distribution_from_csv(uw3,sep=",")
# num=sum([y for x,y in dist])
# print "UW3-->TOTAL:%d,TRAIN:%d,TEST:%d"%(num,0.8*num,0.2*num)
# print dist
# dist2=tool.get_classname_distribution_from_csv(ocrd,sep=",")
# num2=(sum([y for x,y in dist2]))
# print "OCRD-->TOTAL:%d,TRAIN:%d,TEST:%d"%(num2,0.8*num2,0.2*num2)
# print dist2
# #
# cl,cnt=tool.get_class_distribution(ocrd,sep=",")
# print cl
# print cnt
# uw3balanced="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/balanced.csv"
# uw3train="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/holdout/train.csv"
# uw3val="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/holdout/val.csv"
# uw3test="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/holdout/test.csv"
# print tool.get_class_distribution(uw3balanced,sep=" ")
# print tool.get_classname_distribution_from_csv(uw3balanced)
# print tool.get_classname_distribution_from_csv(uw3train)
# print tool.get_classname_distribution_from_csv(uw3val)
#print tool.get_classname_distribution_from_csv(uw3test)
# ocrd="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_csv.csv"
# print tool.get_classname_distribution_from_csv(ocrd,sep=",")
g=csv_feature_generator()
# # img_folder="/home/andri/Documents/s2/5/master_arbeit/dataset/UWIII/ZONE_NO_REDUNDANCY"
# # csv_folder="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/dataset/histograms/orisize_rlbwxyh_full_class.csv"
# # # #img_folder="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/hase/A00A"
# # # #csv_folder="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/hase"
uw3_aug="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented"
uw3_aug_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented.csv"
uw3_aug_for_cnn="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/uw3_aug_for_cnn.csv"
uw3_aug_rlbwxyh="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/uw3_aug_rlbwxyh.csv"
uw3_aug_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/holdout/train.csv"
uw3_aug_val_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/holdout/val_test.csv"
uw3_aug_rlbwxyh_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/uw3_aug_rlbwxyh_train.csv"
uw3_aug_rlbwxyh_val_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/uw3_aug_rlbwxyh_val_test.csv"
#g.convert_images_csv_to_rlbwxyh_csv(csv_file=uw3_aug_train_csv,csv_path=uw3_aug_rlbwxyh_val_test_csv)

res_aug_rlbwxyh="/home/andri/Documents/s2/5/master_arbeit/app/logical_labeling/test_model/res_aug_rlbwxh_pasted.csv"
res_aug_rlbwxyh2="/home/andri/Documents/s2/5/master_arbeit/app/logical_labeling/test_model/res_aug_rlbwxh_revised.csv"

df=pd.read_csv(res_aug_rlbwxyh,sep=",",header=None)
npd=np.array(df)
#print len(npd)
# a= npd[0,0]
# print len(a)
# print "____________"
# b= npd[0,1]
# print len(b)
#print npd[:,1][1]
# a=npd[:,0]
# b=npd[:,1]
# i=0
# bb=[]
# for i in range(len(npd)):
#     w=npd[i,0]
#     x=npd[i,1][1]
#     bb.append([w,x])
# #=zip([a,b])
# np.savetxt(res_aug_rlbwxyh2,bb,fmt="%s",delimiter=" ")


tfknn=tf_knn()
uw3_aug_rlbwxyh="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/uw3_aug_rlbwxyh.csv"
uw3_aug_rlbwxyh_train="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/uw3_aug_rlbwxyh_train.csv"
uw3_aug_rlbwxyh_test="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/uw3_aug_rlbwxyh_test.csv"
uw3_aug_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/holdout/train.csv"
uw3_aug_val_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/holdout/val.csv"
uw3_aug_test_csv= "/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/holdout/test.csv"
uw3_aug_val_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/unrelocated/227/augmented/holdout/val_test.csv"
#tool.merge_csvs([uw3_aug_val_csv,uw3_aug_test_csv],uw3_aug_val_test_csv)
#tfknn.knn_test()
uw3_train_csv= tool.get_classname_distribution_from_csv(uw3_aug_train_csv)
num_classes=len(uw3_train_csv)
uw3_mean_pixels=[127,127,127]
# print tool.get_classname_distribution_from_csv(uw3_aug_val_csv)
# print tool.get_classname_distribution_from_csv(uw3_aug_test_csv)
# print tool.get_classname_distribution_from_csv(uw3_aug_val_test_csv)
#df=pd.read_csv(uw3_aug_rlbwxyh,sep=",")
# tfknn.knn_test(train_csv=uw3_aug_train_csv,
#                test_csv=uw3_aug_val_test_csv,
#                num_classes=num_classes,
#                mean_pixels=uw3_mean_pixels)
uw3_ori_hist_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_csv.csv"
uw3_ori_hist_train_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_ori_hist_train.csv"
uw3_ori_hist_test_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_ori_hist_test.csv"

df=pd.read_csv(uw3_ori_hist_csv,sep=",",header=None)
#df=pd.read_csv(uw3_aug_rlbwxyh,sep=",",header=None)
npd=np.array(df)
datas=npd[:]
labels=npd[:,-1]
kfold=KFold(n_splits=5)
# for train_idx, test_idx in kfold.split(labels):
#     train_datas = datas[train_idx[:]]
#     train_labels = labels[train_idx[:]]
#     test_datas = datas[test_idx[:]]
#     test_labels = labels[test_idx[:]]
#     np.savetxt(uw3_ori_hist_train_csv, train_datas, fmt="%s", delimiter=",")
#     np.savetxt(uw3_ori_hist_test_csv, test_datas, fmt="%s", delimiter=",")
#     # np.savetxt(uw3_aug_rlbwxyh_train,zip(train_datas,train_labels),fmt="%s",delimiter=",")
#     # np.savetxt(uw3_aug_rlbwxyh_test, zip(test_datas, test_labels), fmt="%s", delimiter=",")
#     break
#     # for idx in range(0, len(test_labels)):
#     #     pass
# a=[1,2,3,4,5,6,7,8,9]
# npd=np.array(a[0])
# for aa in a:
#     npdd=np.array([aa])
#     npd=np.vstack((npd,npdd))
#
# print npd

# reuw3ori="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_ori_result.csv"
# reuw3ori2="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_ori_result2.csv"
# uw3_dct_csv="/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/uw3_dictionary.csv"
# df=pd.read_csv(reuw3ori2,sep=",",header=None)
# npd=np.array(df)
# uw3_dct=tool.convert_csv_to_dictionary(uw3_dct_csv)
# uw3_dct={v:k for k,v in uw3_dct.items()}
# arr=[]
# # with open(reuw3ori2, "wb") as f:
# #     writer=csv.writer(f)
# #     for row in npd:
# #         tmp=[uw3_dct[row[0]],uw3_dct[row[1]]]
# #         writer.writerow(tmp)
#
# #tool.calculate_f1_measure(npd[:,0],npd[:,-1])
# ocrd_ori_hist="/home/andri/Documents/s2/5/master_arbeit/csv/ocrd/histogram/ocrd_zones_hifreq_histogram_csv.csv"
# uw3_csv="/home/andri/Documents/s2/5/master_arbeit/csv/uw3/histogram/uw3_zones_noredundant_hifreq_histogram_csv.csv"
#
# df=pd.read_csv(uw3_csv,sep=",",header=None)
# print len(df)
# df=pd.read_csv(ocrd_ori_hist,sep=",",header=None)
# print len(df)

# f=open('asd.dat','ab')
# for iind in range(4):
#     a=np.random.rand(10,10)
#     np.savetxt(f,a)
# f.close()

# uw3_aug_rlbwxh_all="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/uw3_aug_rlbwxh_result.csv"
# uw3_aug_train="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_train.csv"
# uw3_aug_test="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_test.csv"
# tool=victorinox()
# tool.generate_balanced_train_val_by_percentage(dataset_csv=uw3_aug_rlbwxh_all,
#                                                train_csv=uw3_aug_rlbwxyh_train,
#                                                test_csv=uw3_aug_rlbwxyh_test,
#                                                train_percentage=0.75,
#                                                test_percentage=0.25,
#                                                val_percentage=0,sep=",",
#                                                class_id_is_label=True)
# print tool.get_classname_distribution_from_csv(uw3_aug_rlbwxyh_train,sep=",",class_id_is_label=True)
# print tool.get_classname_distribution_from_csv(uw3_aug_rlbwxyh_test,sep=",",class_id_is_label=True)
# path="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented"
# i=0
# for dirpath, dirs, files in os.walk(path):
#     if len(files) > 0:
#         for file in files:
#             if str(file).endswith(".bin.png"):
#                 i+=1
# print i
#
# ocrd_aug_rlbwxh_result="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented_rlbwxh_result.csv"
# uw3_aug_rlbwxh_result="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/uw3_aug_rlbwxh_result.csv"
# df=pd.read_csv(uw3_aug_rlbwxh_result,sep=",",header=None)
# npd=np.array(df)
# uniques=[]
# redundant=[]
#
# for i in range(len(npd)):
#     if i>0:
#         if np.array_equal(npd[i],npd[i-1]) and (i%100==99 or i%100==0 or i%100==1):
#             print "INDEX:%d-------------%s ==%s"%(i, npd[i],npd[i-1])
#             redundant.append(npd[i])
#             continue
#
#             # if len(redundant)==10:
#             #     break#continue
#     uniques.append(npd[i])
# print "TOTAL=%d"%len(npd)
# print "REDUNDANT=%d"%len(redundant)
# print "UNIQUE=%d"%len(uniques)
# print "RED+UNI=%d"%(len(redundant)+len(uniques))

# arr=[]
# arr2=[]
# resume =1000
# savebatch=1
# stop=1010
# with open("test.csv","a+") as f:
#     pass
# k=0
# for i in range(1009):
#     if i<resume:
#         k+=1
#         continue
#     arr.append(i)
#     if k==stop:
#         break
#     batch=k+1
#     if batch%savebatch==0 and i!=0:
#         hist2csv = arr[-savebatch:]
#
#             #arr2.append(hist2csv[ii])
#         with open("test.csv", "a+") as f:
#             w=csv.writer(f)
#             for ii in range(savebatch):
#                 w.writerow([hist2csv[ii]])
#     k+=1
# print arr
# print arr2
# print np.array_equal(arr[:1000],arr2)

# total=0
# arr=[]
#
# # for i in range(800000):
# #     try:
# #         arr.append(i)
# #     except Exception,e:
# #         print i
# img="/home/andri/Documents/bilder im deutschland/boarding pass abfahrt nach deutscland/20160401_053600.jpg"
# arr=cv2.imread(img)
# arr2=arr
# for i in range(100):
#     arr2=np.concatenate((arr2,arr),axis=2)
# start=time()
# total=np.sum(arr2)
# end=time()
# print "TOTAL:%f"%total
# print "TIME:%f"%(end-start)
# sh=np.shape(arr2)
# #arr_tf=tf.get_variable(name="arr",shape=[len(arr)],dtype=tf.int32)
# x=tf.placeholder(dtype=tf.int32,shape=[sh[0],sh[1],sh[2]],name="x")
# total_op=tf.keras.backend.sum(x,axis=2)
#
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     sess.run(tf.initialize_all_variables())
#     start=time()
#     #sess.run(tf.initialize_all_variables())
#     total_tf=sess.run(total_op,feed_dict={x:arr2})
#     end=time()
#     print "TOTAL:%f" % total_tf
#     print "TIME:%f" % (end - start)
# uw3_aug_all="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/augmented.csv"
# uw3_aug_train="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/train.csv"
# uw3_aug_test="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/test.csv"
# uw3_aug_rlbwxyh_train="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_train.csv"
# uw3_aug_rlbwxyh_test="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_test.csv"

ocrd_aug_all="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented.csv"
ocrd_aug_train="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented/train.csv"
ocrd_aug_test="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented/test.csv"
ocrd_aug_rlbwxyh_train="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented/ocrd_aug_rlbwxh_train.csv"
ocrd_aug_rlbwxyh_test="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented/ocrd_aug_rlbwxh_test.csv"



#print tool.find_string_in_dataset_csv(uw3_aug_test,sep=" ")
uw3_aug_all="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/augmented.csv"
uw3_aug_train="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/train.csv"
uw3_aug_test="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/test.csv"
uw3_aug_rlbwxyh_train1="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_train1 (copy).csv"
uw3_aug_rlbwxyh_train2="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_train2.csv"
uw3_aug_rlbwxyh_test="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_test.csv"
uw3_aug_rlbwxyh_train3="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_train3.csv"
uw3_aug_rlbwxyh_train4="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_train4.csv"
uw3_aug_rlbwxyh_train="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_train.csv"
uw3_rlbwxyh_classification_result_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_classification_result.csv"
uw3_rlbwxyh_classification_result_converted_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_classification_result_converted.csv"
ocrd_aug_rlbwxyh_classification_result_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented/ocrd_aug_rlbwxh_classification_result.csv"



df=pd.read_csv(ocrd_aug_rlbwxyh_classification_result_csv,sep=",",header=None)
df2=pd.read_csv(ocrd_aug_rlbwxyh_test,sep=",",header=None)
#print len(df)
npd=np.array(df)
uniques=[]
redundant=[]

# for i in range(len(npd)):
#     if i>0:
#         if np.array_equal(npd[i],npd[i-1]):# and (i%100==99 or i%100==0 or i%100==1):
#             print "INDEX:%d-------------%s ==%s"%(i, npd[i],npd[i-1])
#             redundant.append(npd[i])
#             continue
#
#             # if len(redundant)==10:
#             #     break#continue
#     uniques.append(npd[i])
# print "TOTAL TRAIN=%d"%len(df)
# print "TOTAL Run-Length=%d"%len(df2)
# print "REDUNDANT=%d"%len(redundant)
# print "UNIQUE=%d"%len(uniques)
# print "RED+UNI=%d"%(len(redundant)+len(uniques))
# df=pd.read_csv(uw3_aug_test,sep=" ",header=None)
# npd=np.array(df)
# print "TOTAL FILE:%d" %len(npd)
#tool.calculate_f1_measure(npd[:,0],npd[:,-1])
#uw_dict_csv="/home/andri/Documents/s2/5/master_arbeit/app/deep_dsse/uw3_dictionary.csv"
#dct=tool.convert_csv_to_dictionary(uw_dict_csv)
# tool.convert_label_to_id(csvfile=uw3_rlbwxyh_classification_result_csv,
#                          csvtarget=uw3_rlbwxyh_classification_result_converted_csv,
#                          sep=",",dct=dct)
#tool.print_confusion_matrix_from_csv(uw3_rlbwxyh_classification_result_converted_csv,sep=",",class_dict=dct)


# tool.generate_balanced_train_val_by_percentage(dataset_csv=ocrd_aug_all,
#                                                train_csv=ocrd_aug_train,
#                                                test_csv=ocrd_aug_test,
#                                                train_percentage=0.75,
#                                                test_percentage=0.25,
#                                                val_percentage=0,
#                                                sep=",",
#                                                class_id_is_label=True)
uw3_rlbwxyh_classification_result_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/uw3_aug_rlbwxh_classification_result.csv"
ocrd_aug_rlbwxyh_classification_result_csv="/home/andri/Documents/s2/5/master_arbeit/dataset/ocrd/unrelocated/227/augmented/ocrd_aug_rlbwxh_classification_result.csv"

# dff=pd.read_csv(uw3_rlbwxyh_classification_result_csv,sep=",",header=None)
# npd=np.array(dff)
# t=filter(lambda x:x[0]==x[-1],npd)
# print "ACC:%f" %(len(t)/len(npd))

sample="/home/andri/Documents/work/project/minutes/dataset/samples.vec"
#dff=pd.read_csv(uw3_rlbwxyh_classification_result_csv,sep=",",header=None)
dff=pd.read_csv(sample,sep=" ",header=None,skiprows=[0],quoting=csv.QUOTE_NONE, encoding='utf-8')
npd=np.array(dff)
#t=filter(lambda x:x[0]==x[-1],npd)
print "LENGTH of TESTED:%d"%len(npd)
#print "ACC:%f" %(len(t)/len(npd))
print "words:\n%s" %(npd[:,0])

