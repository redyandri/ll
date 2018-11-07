from __future__ import division
import re
import os
from shutil import copytree
from my_tool import my_tool
from numpy import median

p1="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/ZONE"
p2="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/ZONE_NO_REDUNDANCY"
tool=my_tool()
#tool.generate_no_redundancy_dataset(p1,p2)
#tool.get_class_collection(dataset_path=p2)
# high_freq_class=[('text-body.TIF', 3805), ('section-heading.TIF', 750), ('page-number.TIF', 671), ('reference-list-item.TIF', 625), ('caption.TIF', 570), ('page-header.TIF', 541), ('math.TIF', 396), ('list-item.TIF', 369), ('drawing.TIF', 357), ('page-footer.TIF', 313), ('ruling.TIF', 219), ('author.TIF', 151), ('abstract-body.TIF', 150), ('title.TIF', 137), ('affiliation.TIF', 122), ('table.TIF', 105), ('halftone.TIF', 105), ('footnote.TIF', 93), ('reference-heading.TIF', 63), ('list.TIF', 58), ('abstract-heading.TIF', 52), ('article-submission-information.TIF', 47), ('reference-list.TIF', 45), ('keyword-body.TIF', 44), ('not-clear.TIF', 39), ('biography.TIF', 38), ('keyword-heading.TIF', 26), ('drop-cap.TIF', 20)]
# arr=[]
# for h in high_freq_class:
#     arr.append(h[0])
#
# print arr
# a=[7.0,5.9,3.4,8.8,9.4]
# print median(a)
# print str(sum(a)/len(a))
from tool.csv_feature_generator import *
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
g.convert_images_csv_to_rlbwxyh_csv(csv_file=uw3_aug_val_test_csv,csv_path=uw3_aug_rlbwxyh_train_csv)
