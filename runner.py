from KNNClassifier import KNNClassifier
from csv_feature_generator import *
import sys


path="/Users/pratiwiprananingrum/myData/andri/s2/master_arbeit/logical_labeling/dataset/rlbwxh.csv"
knn=KNNClassifier()
knn.classifyKNN(path)

# g=csv_feature_generator()
# # img_folder="/home/andri/Documents/s2/5/master_arbeit/dataset/UWIII/ZONE_HIGH_FREQ32x32"
# # csv_path="/home/andri/Documents/s2/5/master_arbeit/logical_labeling/dataset/rlbwxh.csv"
# img_folder="/Users/pratiwiprananingrum/myData/andri/s2/master_arbeit/dataset/UWIII/ZONE_NO_REDUNDANCY32x32"
# csv_folder="/Users/pratiwiprananingrum/myData/andri/s2/master_arbeit/logical_labeling/dataset/rlbwxh.csv"
# g.convertImageToCSV(img_folder,csv_folder)
